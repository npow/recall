"""Score finished adaptive episodes with the Gemini Vertex judge as they land (pipeline stage).

    python scripts/score_pipeline.py --label full1 --workers 12 --watch

Each cycle: refresh status from the runner, download only rollout.mp4 + run.json for NEW completed
episodes, judge them (released PlayWorld rubric via worldsync.gemini_vertex_judge, gcloud login;
tokens stay in memory), repeat until every planned episode has a score or a recorded failure.
Results: outputs/adaptive/<label>/gemini/episodes/<episode_id>.json (resumable, cached on disk).
Run with the project venv (needs google-genai); the modal CLI must be on PATH.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VOLUME = "worldsync-lingbot-results"
MODAL_PYTHON = os.environ.get("MODAL_PYTHON", "/home/npow/.local/share/uv/tools/modal/bin/python")
DIMENSION = {"gc": "GC", "if": "IF", "insight": "insight", "outsight": "OE"}


def refresh_status(label):
    outcome = subprocess.run([MODAL_PYTHON, str(ROOT / "scripts/adaptive_runner_client.py"), "status", "--label", label],
                             capture_output=True, text=True)
    path = ROOT / "outputs/adaptive" / label / "status.json"
    if outcome.returncode or not path.is_file():
        raise RuntimeError("status refresh failed: " + outcome.stderr[-300:])
    return json.loads(path.read_text())


def fetch_episode(directory, target):
    target.mkdir(parents=True, exist_ok=True)
    for name in ("rollout.mp4", "run.json"):
        if (target / name).is_file():
            continue
        subprocess.run(["modal", "volume", "get", VOLUME, f"{directory}/{name}", str(target) + "/", "--force"],
                       capture_output=True, text=True)
    return (target / "rollout.mp4").is_file()


def episode_id(arm, row):
    return f"{arm}-{row['split']}-{row['task_id']}-seed{row['seed']}"


def pending_rows(status, scored_ids):
    rows = []
    for arm, data in status["arms"].items():
        for row in data["completed"]:
            identifier = episode_id(arm, row)
            if identifier not in scored_ids:
                rows.append((arm, identifier, row))
    return rows


def summarize(label):
    """Per-arm, per-dimension mean of the official Gemini score; overall = equal mean of the dimension means."""
    root = ROOT / "outputs/adaptive" / label / "gemini/episodes"
    table = {}
    for path in sorted(root.glob("*.json")):
        result = json.loads(path.read_text())
        official = (result.get("scores") or {}).get("official")
        if official is None:
            continue
        arm, split = path.stem.split("-")[0], path.stem.split("-")[1]
        table.setdefault(arm, {}).setdefault(DIMENSION[split], []).append(float(official))
    out = {}
    for arm, dims in table.items():
        means = {dimension: sum(values) / len(values) for dimension, values in dims.items()}
        out[arm] = {"dimension_means": means, "counts": {d: len(v) for d, v in dims.items()},
                    "overall_equal_mean": (sum(means.values()) / len(means)) if len(means) == 4 else None}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--download-workers", type=int, default=6)
    parser.add_argument("--watch", action="store_true", help="loop until every planned episode is scored")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--summarize", action="store_true", help="print the table and exit")
    parser.add_argument("--dry-run", action="store_true", help="download and build prompts; no Gemini calls")
    parser.add_argument("--no-refresh", action="store_true",
                        help="read the last saved status.json instead of asking the (possibly stopped) runner app")
    args = parser.parse_args()
    base = ROOT / "outputs/adaptive" / args.label
    if args.summarize:
        print(json.dumps(summarize(args.label), indent=1))
        return
    from worldsync import gemini_vertex_judge
    plan = json.loads((base / "plan.json").read_text())
    expected = len(plan["arms"]) * len(plan["seeds"]) * len(plan["cases"])
    while True:
        status = (json.loads((base / "status.json").read_text()) if args.no_refresh else refresh_status(args.label))
        scored = {p.stem for p in (base / "gemini/episodes").glob("*.json")
                  if not json.loads(p.read_text()).get("error")} if (base / "gemini/episodes").is_dir() else set()
        todo = pending_rows(status, scored)
        failed = sum(len(d["failed"]) for d in status["arms"].values())
        print(json.dumps({"event": "cycle", "completed": sum(d["completed_count"] for d in status["arms"].values()),
                          "failed": failed, "scored": len(scored), "to_score": len(todo), "expected": expected}), flush=True)
        if todo:
            with concurrent.futures.ThreadPoolExecutor(args.download_workers) as pool:
                ok = list(pool.map(lambda item: fetch_episode(item[2]["result_directory"], base / "episodes" / item[1]), todo))
            rows = []
            for (arm, identifier, row), fetched in zip(todo, ok):
                if not fetched:
                    continue
                split, task = row["split"], row["task_id"]
                rows.append({"episode_id": identifier, "split": split, "task_id": task,
                             "video": str((base / "episodes" / identifier / "rollout.mp4").resolve()),
                             "reference_image": str((ROOT / "external/PlayWorld/data" / gemini_vertex_judge.load_dataset_task(
                                 ROOT / "external/PlayWorld", split, task)["image_path"]).resolve()),
                             "world_model": "LingBot-World2 14B causal-fast"})  # neutral: never reveals the arm
            if rows:
                manifest = base / "gemini" / f"manifest-{int(time.time())}.jsonl"
                manifest.parent.mkdir(parents=True, exist_ok=True)
                manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
                try:
                    gemini_vertex_judge.run_manifest(manifest, base / "gemini", playworld=ROOT / "external/PlayWorld",
                                                     workers=args.workers, dry_run=args.dry_run)
                except RuntimeError as error:
                    if "No usable Google credentials" not in str(error):
                        raise
                    # gcloud session expired (reauth needs an interactive login): wait and retry, never crash
                    print(json.dumps({"event": "waiting_for_credentials", "error": str(error)[:160]}), flush=True)
                    time.sleep(120)
                    continue
        finished = len(scored) + len(todo) + failed
        if not args.watch or (not todo and len(scored) + failed >= expected):
            break
        time.sleep(args.interval)
    print(json.dumps(summarize(args.label), indent=1))


if __name__ == "__main__":
    main()
