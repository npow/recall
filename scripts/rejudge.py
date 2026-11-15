"""Re-judge saved rollout videos with a chosen judge, in parallel (API-bound, no GPU).

    python scripts/rejudge.py --label v6 --provider gemini --repeats 1 \
        --videos GC007=path.mp4 IF001=path2.mp4 --output outputs/rejudge/v6-gemini.json
Case ids are looked up in the PlayWorld data (gc/if/insight/outsight).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "external/PlayWorld/data"


def locate(task_id):
    for split in ("gc", "if", "insight", "outsight"):
        for record in json.loads((DATA / split / "data.json").read_text()):
            if record["task_id"] == task_id:
                return split, record
    raise KeyError(task_id)


def judge(job):
    label, task_id, video, repeat, provider, model, directory = job
    split, record = locate(task_id)
    out = directory / f"{label}-{task_id}-r{repeat}.json"
    if not out.exists():
        command = [sys.executable, "-W", "ignore", "-m", "worldsync.judge", "--provider", provider,
            "--dataset", str(DATA / split / "data.json"), "--task-id", task_id, "--video", str(video),
            "--reference-image", str(DATA / record["image_path"]),
            "--world-model", "LingBot-World2 14B causal-fast stock", "--output", str(out)]
        if model:
            command += ["--model", model]
        subprocess.run(command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        data = json.loads(out.read_text())
        return task_id, repeat, data.get("reported_score"), (data.get("gate") or {}).get("verdict")
    except Exception:
        return task_id, repeat, None, "judge_failed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--provider", choices=["anthropic", "gemini"], default="gemini")
    parser.add_argument("--model")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--videos", nargs="+", required=True, metavar="TASK=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output.parent / f"{args.output.stem}-raw"
    directory.mkdir(parents=True, exist_ok=True)
    jobs = [(args.label, item.split("=")[0], Path(item.split("=", 1)[1]), r, args.provider, args.model, directory)
            for item in args.videos for r in range(args.repeats)]
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(judge, jobs))
    cases = {}
    for task, repeat, score, verdict in rows:
        cases.setdefault(task, []).append({"repeat": repeat, "score": score, "gate": verdict})
    for task, items in cases.items():
        scores = [i["score"] for i in items if i["score"] is not None]
        print(task, [i["score"] for i in items], [i["gate"] for i in items])
    means = [statistics.mean([i["score"] for i in items if i["score"] is not None]) for items in cases.values()
             if any(i["score"] is not None for i in items)]
    summary = {"label": args.label, "provider": args.provider, "cases": cases,
        "mean_of_case_means": round(statistics.mean(means), 3) if means else None, "n_scored_cases": len(means), "n_cases": len(cases)}
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print("mean", summary["mean_of_case_means"], "scored", len(means), "/", len(cases))


if __name__ == "__main__":
    main()
