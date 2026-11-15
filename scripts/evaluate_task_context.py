"""Score the frozen18/4 versus22/8 task cohort, retaining missing pairs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_protected_recent import fetch, compare
from worldsync.lingbot_paths import PATHS

LABELS = ("reference18_read4", "candidate22_read8")


def evaluate(case):
    sources = [ROOT / "outputs/lingbot-reproduction" / f"task-context-validation-{case['task_id']}-seed{case['seed']}-{label}.json"
               for label in LABELS]
    if not all(source.exists() for source in sources):
        return {**case, "status": "generation_pending"}
    output = ROOT / "reproduction/task-context-validation" / case["task_id"]
    scores = {}
    try:
        tasks = json.loads((ROOT / "external/PlayWorld/data/gc/data.json").read_text())
        task = next(row for row in tasks if row["task_id"] == case["task_id"])
        for label, source in zip(LABELS, sources):
            run = json.loads(source.read_text())["run"]
            if (run["task_id"] != case["task_id"] or run["seed"] != case["seed"]
                    or run["initial_image_sha256"] != case["image_sha256"]
                    or run["image_caption_sha256"] != case["caption_sha256"]
                    or run["fixed_actions"] != PATHS[case["path"]]):
                raise ValueError("Generated input identity does not match frozen case")
            fetch(label, source, output)
        comparison = compare(LABELS, True, output)
        for label in LABELS:
            video = output / label / "rollout.mp4"
            score_path = output / label / "task-score.json"
            video_hash = hashlib.sha256(video.read_bytes()).hexdigest()
            if score_path.exists() and json.loads(score_path.read_text())["video_sha256"] != video_hash:
                raise ValueError("Cached task score belongs to another video")
            if not score_path.exists() or json.loads(score_path.read_text()).get("reported_score") is None:
                command = [sys.executable, "-m", "worldsync.judge", "--dataset", str(ROOT / "external/PlayWorld/data/gc/data.json"),
                           "--task-id", case["task_id"], "--video", str(video),
                           "--reference-image", str(ROOT / "external/PlayWorld/data" / task["image_path"]),
                           "--world-model", "LingBot-World2 local controlled trial", "--output", str(score_path)]
                log_path = ROOT / "outputs/lingbot-reproduction" / f"task-context-score-{case['task_id']}-{label}.log"
                with log_path.open("w") as log:
                    subprocess.run(command, check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            score = json.loads(score_path.read_text())
            if score.get("reported_score") is None:
                raise ValueError("Released-rubric score missing")
            scores[label] = {"score": score["reported_score"], "gate": score["gate"],
                             "judge_model": score["judge_model"], "video_sha256": score["video_sha256"]}
        return {**case, **comparison, "status": "completed", "task_scores": scores}
    except Exception as error:
        return {**case, "status": "incomplete_or_failed", "error": f"{type(error).__name__}: {error}", "partial_scores": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    plan = json.loads((ROOT / "reproduction/task-context-validation-plan.json").read_text())
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(evaluate, plan["cases"]))
    complete = len(results) == len(plan["cases"]) and all(row["status"] == "completed" for row in results)
    summary = {"scope": plan["scope"], "complete": complete, "declared_pairs": len(plan["cases"]),
               "results": results, "means": None, "quarantined_task_not_counted": "GC013",
               "caveat": plan["caveat"]}
    if complete:
        summary["means"] = {label: sum(row["task_scores"][label]["score"] for row in results) / len(results) for label in LABELS}
    output = ROOT / "reproduction/task-context-validation"
    output.mkdir(exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
