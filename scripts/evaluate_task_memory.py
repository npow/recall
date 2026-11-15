"""Evaluate the entire declared task-matched cohort, including failures."""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_context_validation import evaluate


def evaluate_task(case):
    output = ROOT / "reproduction/task-memory-benchmark"
    try:
        audit = json.loads((ROOT / "reproduction/task-input-audit.json").read_text())["cases"][case["task_id"]]
        if audit["status"] != "valid":
            raw = {}
            directory = output / f"{case['task_id']}-seed{case['seed']}-{case['path']}"
            for mode in ("stock", "retrieval"):
                score_path = directory / f"{mode}-task-score.json"
                if score_path.exists():
                    raw[mode] = json.loads(score_path.read_text()).get("reported_score")
            return {**case, "status": "invalid_input", "reason": audit, "raw_invalid_scores_not_in_mean": raw}
        result = evaluate(case, output, judge=True, memory=True, input_prefix="task-benchmark")
        directory = output / f"{case['task_id']}-seed{case['seed']}-{case['path']}"
        result["local_task_scores"] = {}
        tasks = json.loads((ROOT / "external/PlayWorld/data/gc/data.json").read_text())
        task = next(record for record in tasks if record["task_id"] == case["task_id"])
        for mode in ("stock", "retrieval"):
            score_path = directory / f"{mode}-task-score.json"
            if score_path.exists():
                prior = json.loads(score_path.read_text())
                if prior["video_sha256"] != hashlib.sha256((directory / f"{mode}.mp4").read_bytes()).hexdigest():
                    raise ValueError("Cached task score belongs to a different video")
            if not score_path.exists() or json.loads(score_path.read_text()).get("reported_score") is None:
                command = [sys.executable, "-m", "worldsync.judge", "--dataset", str(ROOT / "external/PlayWorld/data/gc/data.json"),
                           "--task-id", case["task_id"], "--video", str(directory / f"{mode}.mp4"),
                           "--reference-image", str(ROOT / "external/PlayWorld/data" / task["image_path"]),
                           "--world-model", "LingBot-World2 local controlled trial", "--output", str(score_path)]
                with (ROOT / "outputs/lingbot-reproduction" / f"task-score-{case['task_id']}-{mode}.log").open("w") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            score = json.loads(score_path.read_text())
            if score["reported_score"] is None:
                raise ValueError(f"Missing released-rubric effective score: {mode}")
            result["local_task_scores"][mode] = {
                "score": score["reported_score"], "trajectory_gate": score["gate"],
                "judge_model": score["judge_model"], "video_sha256": score["video_sha256"],
            }
        result["status"] = "completed"
        return result
    except Exception as error:
        return {**case, "status": "incomplete_or_failed", "error": f"{type(error).__name__}: {error}"}


def summarize_results(plan, results):
    complete = [result for result in results if result["status"] == "completed"]
    valid_declared = [case for case in plan["cases"] if case["task_id"] not in plan.get("quarantined_case_ids", [])]
    # No mean until every valid declared pair is scored; quarantine stays visible.
    key = lambda case: (case["task_id"], case.get("seed"), case.get("path"))
    expected = {key(case) for case in valid_declared}
    all_complete = len(complete) == len(expected) and {key(result) for result in complete} == expected
    summary = {"scope": plan["scope"], "declared_pairs_including_quarantine": len(plan["cases"]), "valid_declared_pairs": len(valid_declared),
               "completed_pairs": len(complete), "quarantined_case_ids": plan.get("quarantined_case_ids", []),
               "complete": all_complete, "results": results, "local_cohort_means": None,
               "cohort_definition": "Amended pilot after a verified source input mismatch; quarantined raw results remain visible and are not valid quality scores.",
               "caveat": "Scripted task-matched actions and Claude evaluation differ from the published Agent Player/Gemini protocol. This is not a leaderboard score."}
    if all_complete:
        summary["local_cohort_means"] = {mode: sum(result["local_task_scores"][mode]["score"] for result in complete) / len(complete)
                                         for mode in ("stock", "retrieval")}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--available", action="store_true", help="Score completed pairs while the other declared generations continue")
    args = parser.parse_args()
    plan = json.loads((ROOT / "reproduction/task-memory-benchmark-plan.json").read_text())
    cases = plan["cases"]
    if args.available:
        cases = [case for case in cases if all((ROOT / "outputs/lingbot-reproduction" /
                 f"task-benchmark-{case['task_id']}-{mode}-{case['path']}-seed{case['seed']}.json").exists()
                 for mode in ("stock", "retrieval"))]
    output = ROOT / "reproduction/task-memory-benchmark"
    output.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(evaluate_task, cases))
    summary = summarize_results(plan, results)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
