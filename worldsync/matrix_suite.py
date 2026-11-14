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

ROOT = Path(__file__).resolve().parents[1]


def publish(progress):
    """Publish only the bounded, credential-free progress summary authorized by the user."""
    path = ROOT / "reproduction/matrix-progress.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(progress, indent=2) + "\n")
    temporary.replace(path)
    relative = str(path.relative_to(ROOT))
    subprocess.run(["git", "add", "--", relative], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--", relative], cwd=ROOT).returncode == 0:
        return
    subprocess.run(["git", "commit", "--only", "-m", "Update stock Matrix reproduction progress", "--", relative], cwd=ROOT, check=True)
    subprocess.run(["git", "push", "origin", "HEAD"], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "external/PlayWorld/data")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/matrix-reproduction/suite")
    parser.add_argument("--max-chunks", type=int, default=25)
    parser.add_argument("--smoke", action="store_true", help="Four explicit integration cases, one per split; no overall score")
    parser.add_argument("--publish-status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
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
        if progress["protocol_id"] != PROTOCOL_ID or progress["max_chunks"] != args.max_chunks or progress["selected_cases"] != cases:
            parser.error("existing suite configuration differs; choose a new output directory")
    else:
        progress = {"protocol_id": PROTOCOL_ID, "max_chunks": args.max_chunks,
            "selected_cases": cases, "runs": [], "scope": "four-case smoke" if args.smoke else "full 171-case generation reconstruction",
            "published_score_reproduced": False, "hillclimbing_allowed": False, "status": "prepared"}

    def save():
        counts = Counter(run["split"] for run in progress["runs"] if run["status"] == "completed")
        progress["completed_by_split"] = {split: counts[split] for split in ("gc", "if", "insight", "outsight")}
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2) + "\n")
        temporary.replace(manifest_path)
        if args.publish_status and not args.dry_run:
            # No raw errors, API metadata, environment variables, prompts or absolute paths.
            publish({key: progress[key] for key in ("protocol_id", "max_chunks", "scope", "published_score_reproduced",
                "hillclimbing_allowed", "status", "completed_by_split", "runs")})
    save()
    if args.dry_run:
        print(json.dumps({"status": "prepared", "cases": len(cases), "api_calls": 0}))
        return
    progress["status"] = "running"
    save()
    for case in cases:
        split, task_id = case["split"], case["task_id"]
        prior = [run for run in progress["runs"] if (run["split"], run["task_id"]) == (split, task_id)]
        if any(run["status"] == "completed" for run in prior):
            continue
        attempt = len(prior) + 1
        directory = output / f"{split}-{task_id}-attempt{attempt}"
        run = {"split": split, "task_id": task_id, "attempt": attempt, "status": "running"}
        progress["runs"].append(run)
        save()
        started = time.monotonic()
        with (output / f"{split}-{task_id}-attempt{attempt}.log").open("w") as log:
            child = subprocess.run([sys.executable, "-u", "-m", "worldsync.matrix_baseline",
                "--data-root", str(args.data_root.resolve()), "--split", split, "--task-id", task_id,
                "--max-chunks", str(args.max_chunks), "--output", str(directory)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
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
        save()
    progress["status"] = "generation_complete_scoring_unverified"
    save()
    print(json.dumps({"status": progress["status"], "completed_by_split": progress["completed_by_split"]}))


if __name__ == "__main__":
    main()
