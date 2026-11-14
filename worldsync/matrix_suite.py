"""Run/resume the stock Matrix suite, retaining failures and reporting all coverage."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

from worldsync.matrix_baseline import PROTOCOL_ID
from worldsync.repro_assets import inventory_cases
from worldsync.coverage import diagnostic_summary

ROOT = Path(__file__).resolve().parents[1]


def publish(progress, filename="matrix-progress.json"):
    """Publish only the bounded, credential-free progress summary authorized by the user."""
    from worldsync.progress import publish as publish_file
    publish_file(progress, filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["matrix", "lingbot"], default="matrix")
    parser.add_argument("--data-root", type=Path, default=ROOT / "external/PlayWorld/data")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--smoke", action="store_true", help="Four explicit integration cases, one per split; no overall score")
    parser.add_argument("--publish-status", action="store_true")
    parser.add_argument("--judge", action="store_true", help="Claude diagnostics with released rubric; never official Gemini reproduction")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_chunks is None:
        args.max_chunks = 40 if args.model == "lingbot" else 25
    if args.model == "lingbot":
        from worldsync.lingbot_rollout import PROTOCOL_ID as protocol_id
        generator_module, model_name = "worldsync.lingbot_rollout", "LingBot-World2 14B causal-fast stock"
        progress_file = "lingbot-progress.json"
    else:
        protocol_id, generator_module, model_name = PROTOCOL_ID, "worldsync.matrix_baseline", "Matrix-Game 3.0 stock"
        progress_file = "matrix-progress.json"
    if args.output is None:
        args.output = ROOT / "outputs" / ("lingbot-reproduction" if args.model == "lingbot" else "matrix-reproduction") / "suite"
    if not 1 <= args.max_chunks <= 40:
        parser.error("max chunks must be in [1,40]")
    cases = inventory_cases(args.data_root.resolve())
    if args.smoke:
        selected = {("gc", "GC007"), ("if", "IF001"), ("insight", "OE001"), ("outsight", "OE014")}
        cases = [case for case in cases if (case["split"], case["task_id"]) in selected]
    else:
        # Start with the public Matrix reference case without dropping/repeating others.
        cases.sort(key=lambda case: (case["split"] != "gc" or case["task_id"] != "GC007", case["split"], case["task_id"]))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "suite.json"
    if manifest_path.exists():
        progress = json.loads(manifest_path.read_text())
        if progress["protocol_id"] != protocol_id or progress["max_chunks"] != args.max_chunks or progress["selected_cases"] != cases or progress.get("judge_enabled", False) != args.judge:
            parser.error("existing suite configuration differs; choose a new output directory")
    else:
        progress = {"protocol_id": protocol_id, "max_chunks": args.max_chunks,
            "selected_cases": cases, "runs": [], "scope": "four-case smoke" if args.smoke else "full 171-case generation reconstruction",
            "judge_enabled": args.judge, "judge_protocol": "released-rubric-claude-opus-5-5" if args.judge else None,
            "published_score_reproduced": False, "hillclimbing_allowed": False, "status": "prepared"}

    def save():
        progress["world_model"] = model_name
        counts = Counter(run["split"] for run in progress["runs"] if run["status"] == "completed")
        progress["completed_by_split"] = {split: counts[split] for split in ("gc", "if", "insight", "outsight")}
        progress["diagnostic_scores"] = diagnostic_summary(cases, progress["runs"])
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2) + "\n")
        temporary.replace(manifest_path)
        if args.publish_status and not args.dry_run:
            # No raw errors, API metadata, environment variables, prompts or absolute paths.
            publish({key: progress[key] for key in ("protocol_id", "max_chunks", "scope", "published_score_reproduced",
                "hillclimbing_allowed", "status", "completed_by_split", "runs", "judge_protocol", "diagnostic_scores", "world_model")}, progress_file)
    save()
    if args.dry_run:
        print(json.dumps({"status": "prepared", "cases": len(cases), "api_calls": 0}))
        return
    progress["status"] = "running"
    save()
    for case in cases:
        split, task_id = case["split"], case["task_id"]
        prior = [run for run in progress["runs"] if (run["split"], run["task_id"]) == (split, task_id)]
        if any(run["status"] == "completed" and (not args.judge or run.get("judge_status") == "completed") for run in prior):
            continue
        run = next((item for item in prior if item["status"] == "completed"), None)
        if run is not None:
            attempt = run["attempt"]
            directory = output / f"{split}-{task_id}-attempt{attempt}"
        else:
            attempt = len(prior) + 1
            directory = output / f"{split}-{task_id}-attempt{attempt}"
            run = {"split": split, "task_id": task_id, "attempt": attempt, "status": "running"}
            progress["runs"].append(run)
            save()
            started = time.monotonic()
            with (output / f"{split}-{task_id}-attempt{attempt}.log").open("w") as log:
                chunk_budget = 60 if args.model == "lingbot" and split == "insight" and args.max_chunks == 40 and not args.smoke else args.max_chunks
                child = subprocess.run([sys.executable, "-u", "-m", generator_module,
                    "--data-root", str(args.data_root.resolve()), "--split", split, "--task-id", task_id,
                    "--max-chunks", str(chunk_budget), "--output", str(directory)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            result_path = directory / "run.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            run.update(status="completed" if child.returncode == 0 and result.get("status") == "completed" else "failed",
                elapsed_s=round(time.monotonic() - started, 2), exit_code=child.returncode,
                generated_frames=result.get("generated_frames", 0), generated_seconds=result.get("generated_seconds", 0),
                termination=result.get("termination"), chunks=len(result.get("chunks", [])))
            if run["status"] != "completed":
                progress["status"] = "stopped_on_failure"
                save()
                raise SystemExit(f"Stock run failed at {split}/{task_id}; preserved log and artifacts. No cases silently dropped.")
        if args.judge:
            run["judge_status"] = "running"
            save()
            records = json.loads((args.data_root / split / "data.json").read_text())
            task = next(record for record in records if record["task_id"] == task_id)
            judge_attempt = run.get("judge_attempts", 0) + 1
            run["judge_attempts"] = judge_attempt
            judge_path = directory / "judge.json"
            if judge_path.exists():
                judge_path.rename(directory / f"judge-prior-{judge_attempt - 1}.json")
            with (output / f"{split}-{task_id}-attempt{attempt}-judge-{judge_attempt}.log").open("w") as log:
                judge = subprocess.run([sys.executable, "-u", "-m", "worldsync.judge",
                    "--dataset", str((args.data_root / split / "data.json").resolve()), "--task-id", task_id,
                    "--video", str(directory / "rollout.mp4"), "--reference-image", str((args.data_root / task["image_path"]).resolve()),
                    "--world-model", model_name, "--output", str(directory / "judge.json")],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            judge_path = directory / "judge.json"
            judged = json.loads(judge_path.read_text()) if judge_path.exists() else {}
            score = judged.get("reported_score")
            if judge.returncode or isinstance(score, bool) or not isinstance(score, (int, float)) or not 1 <= score <= 5:
                run["judge_status"] = "failed"
                progress["status"] = "stopped_on_judge_failure"
                save()
                raise SystemExit(f"Judge failed at {split}/{task_id}; no benchmark score inferred.")
            run.update(judge_status="completed", diagnostic_score=score,
                gate_verdict=(judged.get("gate") or {}).get("verdict"))
        save()
    progress["status"] = "generation_and_claude_diagnostics_complete_not_official" if args.judge else "generation_complete_scoring_unverified"
    save()
    print(json.dumps({"status": progress["status"], "completed_by_split": progress["completed_by_split"]}))


if __name__ == "__main__":
    main()
