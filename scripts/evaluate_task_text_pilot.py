"""Evaluate every declared IF text-conditioning episode with the released Gemini verifier."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worldsync.gemini_vertex_judge import create_vertex_client, judge_episode

DEFAULT_OUTPUT = ROOT / "reproduction/task-text-pilot"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.directory.resolve()
    plan = json.loads((output / "plan.json").read_text())
    jobs = [json.loads(line) for line in (output / "jobs.jsonl").read_text().splitlines()]
    expected = {(case["task_id"], case.get("seed", plan.get("seed")), arm)
                for case in plan["cases"] for arm in plan["arms"]}
    declared = []
    for job in jobs:
        argv = job["argv"]
        declared.append((argv[argv.index("--task-id") + 1],
                         int(argv[argv.index("--seed") + 1]),
                         argv[argv.index("--text-prompt") + 1]))
    if (len(expected) != len(plan["cases"]) * len(plan["arms"])
            or len(jobs) != len(expected) or set(declared) != expected
            or len({job["id"] for job in jobs}) != len(jobs)):
        raise ValueError("Jobs must cover every frozen case/seed/arm exactly once")
    bundle = json.loads((output / "results.json").read_text())
    outcomes = {row["id"]: row for row in bundle["results"]}
    if len(outcomes) != len(bundle["results"]) or set(outcomes) != {job["id"] for job in jobs}:
        raise ValueError("Returned jobs do not cover the exact declared pilot")
    if not json.loads((ROOT / "reproduction/state.json").read_text())["gemini_calls_authorized"]:
        raise ValueError("Gemini evaluation is not authorized")
    data = json.loads((ROOT / "external/PlayWorld/data/if/data.json").read_text())
    tasks = {task["task_id"]: task for task in data}
    for case in plan["cases"]:
        task = tasks[case["task_id"]]
        if hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest() != case["task_sha256"]:
            raise ValueError("Frozen task data changed")
    client = create_vertex_client()

    def evaluate(job):
        result = outcomes[job["id"]]
        argv = job["argv"]
        task_id = argv[argv.index("--task-id") + 1]
        mode = argv[argv.index("--text-prompt") + 1]
        seed = int(argv[argv.index("--seed") + 1])
        case = next(case for case in plan["cases"] if case["task_id"] == task_id
                    and case.get("seed", plan.get("seed")) == seed)
        row = {"id": job["id"], "task_id": task_id, "mode": mode, "seed": seed,
               "partition": case.get("partition", "development")}
        run = result.get("run") or {}
        if result.get("error") or run.get("status") != "completed":
            return {**row, "status": "generation_failed", "score": 1.0,
                    "error": result.get("error") or run.get("error"),
                    "failure_policy": "Retained model/transport failure, not excluded from arm mean"}
        if (run["task_id"] != task_id or run["seed"] != seed or run["split"] != "if"
                or run["text_prompt_mode"] != mode or run["native_fixed_controls"] or run["fixed_actions"]
                or run["initial_image_sha256"] != case["image_sha256"]
                or run["local_attn_size"] != 18 or run["sink_size"] != 6
                or run["max_chunks"] != 40 or not run["player_streaming"]
                or any(action.get("player_response_pending") for action in run["actions"])):
            raise ValueError("Episode differs from the declared text pilot")
        directory = output / job["id"]
        directory.mkdir(exist_ok=True)
        (directory / "run.json").write_text(json.dumps(run, indent=2) + "\n")
        (directory / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        video = directory / "rollout.mp4"
        if not video.exists():
            subprocess.run(["modal", "volume", "get", "--profile", "nissan-pow", "worldsync-lingbot-results",
                            result["result_directory"] + "/rollout.mp4", str(video)], check=True)
        score_path = directory / "gemini-score.json"
        previous = json.loads(score_path.read_text()) if score_path.exists() else None
        digest = hashlib.sha256(video.read_bytes()).hexdigest()
        if previous and previous["video_sha256"] != digest:
            raise ValueError("Cached score belongs to a different video")
        score = previous or judge_episode(client, ROOT / "external/PlayWorld", tasks[task_id], video,
            ROOT / "external/PlayWorld/data" / tasks[task_id]["image_path"],
            world_model="LingBot-World2 task conditioning trial", cache_dir=output / "cache")
        value = (score.get("scores") or {}).get("official")
        if value is None or not math.isfinite(value) or not 1 <= value <= 5:
            raise ValueError("Gemini returned no valid official score")
        temporary = score_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(score, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(score_path)
        return {**row, "status": "scored", "score": value, "gate": score["gate"],
                "generated_seconds": run["generated_seconds"], "agent_interactions": run["agent_interactions"],
                "termination": run["termination"], "video_sha256": digest,
                "judge_model": score["judge_model"], "hardware": run["hardware"],
                "interaction_quality": (score.get("rubric") or {}).get("if_interaction_quality_check")}

    def guarded(job):
        try:
            return evaluate(job)
        except Exception as error:
            return {"id": job["id"], "status": "evaluation_failed", "score": None,
                    "error": f"{type(error).__name__}: {str(error)[:300]}"}
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(guarded, jobs))
    complete = all(row["status"] in {"scored", "generation_failed"} for row in results)
    report = {"scope": plan["scope"], "complete": complete, "declared_episodes": len(jobs),
              "results": results, "means": None,
              "caveat": "Small declared IF cohort with Gemini preview alias; not a four-dimension overall. Development and confirmation partitions remain separate."}
    if complete:
        report["means"] = {mode: sum(row["score"] for row in results if row["mode"] == mode) / len(plan["cases"])
                           for mode in plan["arms"]}
        report["means_by_partition"] = {partition: {mode: sum(row["score"] for row in results
            if row["mode"] == mode and row["partition"] == partition) / sum(case.get("partition", "development") == partition
            for case in plan["cases"]) for mode in plan["arms"]}
            for partition in sorted({case.get("partition", "development") for case in plan["cases"]})}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
