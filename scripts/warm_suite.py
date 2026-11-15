"""Run adaptive LingBot cases in one warm process: load the 14B model once, retry failures, resume.

    python scripts/warm_suite.py --cases gc:GC007 if:IF001 --out outputs/final/run1 [--seed 42] [-- --rollout-flag ...]

Per-case output matches the suite layout (<out>/<split>-<task>-attempt1/ with rollout.mp4, run.json), so
scripts/rejudge.py can score it. A case whose run.json says completed is skipped on restart. A failed case
(for example a CUDA OOM while the GPU is shared) is retried after a pause and is never silently dropped:
it stays in summary.json with its error. No judging here; judge afterwards with scripts/rejudge.py.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def completed(directory):
    try:
        return json.loads((directory / "run.json").read_text()).get("status") == "completed" and (directory / "rollout.mp4").exists()
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", required=True, metavar="SPLIT:TASK")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--pause", type=int, default=90, help="seconds to wait before retrying a failed case")
    parser.add_argument("rollout_flags", nargs="*", help="extra lingbot_rollout flags after --")
    args = parser.parse_args()
    from worldsync import lingbot_rollout
    from worldsync.lingbot_runtime import WarmRuntime
    runtime = WarmRuntime()
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    started = time.monotonic()
    for case in args.cases:
        split, task = case.split(":")
        directory = args.out / f"{split}-{task}-attempt1"
        if completed(directory):
            summary.setdefault(case, {"status": "completed", "note": "resumed"})
            continue
        for attempt in range(1, args.retries + 1):
            if directory.exists():
                directory.rename(args.out / f"{split}-{task}-failed{int(time.time())}")
            argv = ["--data-root", str(ROOT / "external/PlayWorld/data"), "--split", split, "--task-id", task,
                    "--max-chunks", str(60 if split == "insight" else 40), "--seed", str(args.seed),
                    "--output", str(directory), *args.rollout_flags]
            t0, error = time.monotonic(), None
            try:
                lingbot_rollout.main(argv, runtime=runtime)
            except SystemExit as request:
                error = f"rejected (exit {request.code})"
                break  # a bad flag will not fix itself on retry
            except Exception as failure:
                error = f"{type(failure).__name__}: {str(failure)[:300]}"
            runtime.between_jobs()
            if error is None and completed(directory):
                summary[case] = {"status": "completed", "attempts": attempt, "seconds": round(time.monotonic() - t0, 1)}
                break
            summary[case] = {"status": "failed", "attempts": attempt, "error": error or "incomplete run", "seconds": round(time.monotonic() - t0, 1)}
            print(json.dumps({case: summary[case]}), flush=True)
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            time.sleep(args.pause)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"case": case, **summary[case], "elapsed_total_s": round(time.monotonic() - started)}), flush=True)
    failed = [c for c, v in summary.items() if v["status"] != "completed"]
    print(json.dumps({"done": True, "failed": failed, "total": len(summary)}))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
