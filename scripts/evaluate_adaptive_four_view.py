"""Score all declared adaptive-pilot attempts, preserving failures and coverage."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def evaluate(seed, mode, plan, plan_hash):
    source = ROOT / "outputs/lingbot-reproduction" / f"adaptive-four-view-{mode}-seed{seed}.json"
    result = {"seed": seed, "mode": mode, "status": "generation_pending"}
    if not source.exists():
        return result
    manifest = json.loads(source.read_text())
    run = manifest.get("run")
    if manifest["frozen_plan_sha256"] != plan_hash:
        raise ValueError("Pilot plan differs from the generated attempt")
    if manifest["returncode"] != 0 or not run or run["status"] != "completed":
        return {**result, "status": "generation_failed", "result_directory": manifest["result_directory"],
                "error": (run or {}).get("error", "generation process failed"), "effective_score": None}
    if (run["task_id"] != plan["case"]["task_id"] or run["seed"] != seed
            or run["protocol_id"] != plan["protocol_id"] or run["native_fixed_controls"]
            or run["initial_image_sha256"] != plan["case"]["image_sha256"]
            or run["image_caption_sha256"] != plan["case"]["caption_sha256"]
            or run["fixed_actions"] is not None or run["agent_interactions"] < 1
            or run["pose_retrieval"] != (mode == "memory")):
        raise ValueError("Attempt identity does not match the adaptive protocol")
    directory = ROOT / "reproduction/adaptive-four-view" / f"{mode}-seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "run.json"
    if target.exists() and json.loads(target.read_text())["result_directory"] != manifest["result_directory"]:
        raise ValueError("Preserve the earlier attempt before evaluating another run")
    shutil.copy2(source, target)
    video = directory / "rollout.mp4"
    if not video.exists():
        subprocess.run(["modal", "volume", "get", "--profile", "nissan-pow", "worldsync-lingbot-results",
                        manifest["result_directory"] + "/rollout.mp4", str(video)], check=True)
    score_path = directory / "task-score.json"
    video_hash = hashlib.sha256(video.read_bytes()).hexdigest()
    if score_path.exists() and json.loads(score_path.read_text())["video_sha256"] != video_hash:
        raise ValueError("Cached adaptive score belongs to another video")
    if not score_path.exists() or json.loads(score_path.read_text()).get("reported_score") is None:
        log_path = ROOT / "outputs/lingbot-reproduction" / f"adaptive-four-view-score-{mode}-seed{seed}.log"
        with log_path.open("w") as log:
            subprocess.run([sys.executable, "-m", "worldsync.judge", "--dataset",
                str(ROOT / "external/PlayWorld/data/gc/data.json"), "--task-id", plan["case"]["task_id"],
                "--video", str(video), "--reference-image", str(ROOT / "external/PlayWorld/data/gc/images/GC027.jpg"),
                "--world-model", "LingBot-World2 adaptive local trial", "--output", str(score_path)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    score = json.loads(score_path.read_text())
    if score.get("reported_score") is None:
        raise ValueError("Missing effective rubric score")
    interventions = [row["chunk"] for row in run.get("memory", {}).get("trace", []) if row["attention_replaced"]]
    return {**result, "status": "scored", "effective_score": score["reported_score"],
            "trajectory_gate": score["gate"], "judge_model": score["judge_model"],
            "generated_seconds": run["generated_seconds"], "agent_interactions": run["agent_interactions"],
            "termination": run["termination"], "retrieval_chunks": interventions,
            "memory_intervened": bool(interventions), "result_directory": manifest["result_directory"],
            "video_sha256": video_hash, "hardware": run["hardware"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", action="store_true")
    args = parser.parse_args()
    path = ROOT / "reproduction/adaptive-four-view-plan.json"
    plan = json.loads(path.read_text())
    plan_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    mode = "memory" if args.memory else "stock"
    def guarded(seed):
        try:
            return evaluate(seed, mode, plan, plan_hash)
        except Exception as error:
            return {"seed": seed, "mode": mode, "status": "scoring_failed",
                    "error": f"{type(error).__name__}: {error}", "effective_score": None}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(guarded, plan["seeds"]))
    complete = len(results) == len(plan["seeds"]) and all(row["status"] == "scored" for row in results)
    summary = {"scope": plan["scope"], "mode": mode, "complete": complete, "results": results,
               "frozen_plan_sha256": plan_hash, "mean_effective_score": None,
               "caveat": "Separate30deg/s neutral-bootstrap controller, two seeds, local Claude rubric. Not published-score reproduction."}
    if complete:
        summary["mean_effective_score"] = sum(row["effective_score"] for row in results) / len(results)
    directory = ROOT / "reproduction/adaptive-four-view"
    directory.mkdir(exist_ok=True)
    temporary = directory / f"{mode}-summary.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(directory / f"{mode}-summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
