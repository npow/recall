"""Score every frozen start-anchor replay, retaining failed/missing jobs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reproduction/start-anchor-replay"


def evaluate(job, case, group):
    identifier = job["id"]
    folder = OUTPUT / identifier
    result = {"id": identifier, "status": "generation_pending"}
    run_path = folder / "run.json"
    if not run_path.exists():
        return result
    run = json.loads(run_path.read_text())
    if run["status"] != "completed":
        return {**result, "status": "generation_failed", "error": run.get("error")}
    actions = job["argv"][job["argv"].index("--fixed-actions") + 1:]
    actions = actions[:actions.index("--offload-text-encoder")]
    if (not run["native_fixed_controls"] or run["fixed_actions"] != actions
            or len(run["chunks"]) != 13 or len(run["camera_embedding_audit"]) != 13
            or not all(row["within_tolerance"] for row in run["camera_embedding_audit"])
            or run["initial_image_sha256"] != case["image_sha256"]
            or run["image_caption_sha256"] != case["caption_sha256"]):
        raise ValueError("Replay identity, completion, or native camera audit failed")
    if group is None or group.get("returncode") != 0:
        raise ValueError("No complete group manifest for this replay")
    video = folder / "rollout.mp4"
    if not video.exists():
        subprocess.run(["modal", "volume", "get", "--profile", "nissan-pow", "worldsync-lingbot-results",
                        f"{group['result_directory']}/{identifier}/rollout.mp4", str(video)], check=True)
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    score_path = folder / "task-score.json"
    previous = json.loads(score_path.read_text()) if score_path.exists() else {}
    if previous and previous["video_sha256"] != digest:
        raise ValueError("Cached score belongs to a different replay video")
    if previous.get("reported_score") is None:
        with (ROOT / "outputs/lingbot-reproduction" / f"start-anchor-score-{identifier}.log").open("w") as log:
            subprocess.run([sys.executable, "-m", "worldsync.judge", "--dataset",
                str(ROOT / "external/PlayWorld/data/gc/data.json"), "--task-id", "GC027", "--video", str(video),
                "--reference-image", str(ROOT / "external/PlayWorld/data/gc/images/GC027.jpg"),
                "--world-model", "LingBot-World2 controlled local trial", "--provider", "anthropic",
                "--model", "claude-opus-5-5", "--output", str(score_path)], cwd=ROOT, check=True,
                stdout=log, stderr=subprocess.STDOUT)
    score = json.loads(score_path.read_text())
    if score.get("reported_score") is None:
        raise ValueError("Missing replay score")
    return {**result, "status": "scored", "seed": run["seed"], "score": score["reported_score"],
            "gate": score["gate"], "video_sha256": digest, "judge_model": score["judge_model"],
            "retrieval_chunks": [r["chunk"] for r in run.get("memory", {}).get("trace", []) if r["attention_replaced"]],
            "peak_allocated_bytes": run["peak_allocated_bytes"], "hardware": run["hardware"]}


def main():
    plan = json.loads((ROOT / "reproduction/start-anchor-replay-plan.json").read_text())
    source = json.loads((ROOT / "reproduction/adaptive-four-view/judge-source.json").read_text())
    for name, digest in source["files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen Claude judge source changed: {name}")
    groups = [json.loads(path.read_text()) for path in OUTPUT.glob("group-*.json")]
    def guarded(job):
        try:
            group = next((g for g in groups if job["id"] in g.get("declared_job_ids", [])), None)
            return evaluate(job, plan["case"], group)
        except Exception as error:
            return {"id": job["id"], "status": "failed", "error": f"{type(error).__name__}: {error}"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(guarded, plan["jobs"]))
    complete = len(results) == 6 and all(row["status"] == "scored" for row in results)
    report = {"scope": plan["scope"], "complete": complete, "declared_jobs": 6, "results": results,
              "means": None, "limitation": "Exploratory fixed replay of one failed adaptive task, two seeds; judge variability. Require observed revisit, not just a rubric increase."}
    if complete:
        report["means"] = {mode: sum(r["score"] for r in results if r["id"].endswith("-" + mode)) / 2
                           for mode in ("stock", "four-view", "start-anchor")}
    temporary = OUTPUT / "scores-summary.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(OUTPUT / "scores-summary.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
