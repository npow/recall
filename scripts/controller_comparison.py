"""Paired validation: Codex's canonical controller against the legacy player, same commit, seed, GPU and flags.

    python scripts/controller_comparison.py --cases gc:GC007 if:IF001 ... --out outputs/controller-comparison [--dry-run]

Both arms use `lingbot_released_baseline.generation_argv` (task objective as text, stock 18/6 cache, 60 deg/s, text-encoder
offload, seed). Arm C runs `lingbot_released_baseline.run(..., validate_controller=True)` (canonical timeline + released hold
service); arm L runs the same argv through `lingbot_rollout.main` with the default legacy player. One warm model serves both
arms; arms alternate per case. Completed episodes are skipped on restart; failures are retried and kept in summary.json.
Not benchmark evidence.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def episode_done(directory):
    run = directory / "run.json"
    try:
        return json.loads(run.read_text()).get("status") == "completed" and (directory / "rollout.mp4").exists()
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", required=True, metavar="SPLIT:TASK")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunks", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--arms", default="CL", help="C = canonical controller, L = legacy player on current main (default both)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from worldsync import lingbot_released_baseline as baseline
    args.out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        for case in args.cases:
            split, task = case.split(":")
            audit = baseline.preflight(split, task, seed=args.seed, chunks=args.chunks)
            print(case, "canonical preflight:", audit["status"])
        return
    from worldsync import lingbot_rollout
    from worldsync.lingbot_runtime import WarmRuntime
    runtime = WarmRuntime()
    summary_path = args.out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for case in args.cases:
        split, task = case.split(":")
        for arm in [a for a in ("C", "L") if a in args.arms]:
            key = f"{arm}:{case}"
            episode = args.out / arm / f"{split}-{task}" / "episode"
            if episode_done(episode):
                summary.setdefault(key, {"status": "completed", "note": "resumed"})
                continue
            for attempt in range(1, args.retries + 2):
                case_dir = episode.parent
                if case_dir.exists():
                    shutil.move(str(case_dir), str(case_dir.with_name(case_dir.name + f"-failed{int(time.time())}")))
                started, error = time.monotonic(), None
                try:
                    if arm == "C":
                        baseline.run(split, task, case_dir, seed=args.seed, chunks=args.chunks,
                                     validate_controller=True, runtime=runtime)
                    else:
                        case_dir.mkdir(parents=True)
                        lingbot_rollout.main(baseline.generation_argv(split, task, episode, seed=args.seed, chunks=args.chunks),
                                             runtime=runtime)
                except SystemExit as request:
                    error = f"rejected (exit {request.code})"
                except Exception as failure:
                    error = f"{type(failure).__name__}: {str(failure)[:300]}"
                runtime.between_jobs()
                ok = error is None and episode_done(episode)
                summary[key] = {"status": "completed" if ok else "failed", "attempts": attempt,
                                "seconds": round(time.monotonic() - started, 1), **({} if ok else {"error": error or "incomplete"})}
                summary_path.write_text(json.dumps(summary, indent=2) + "\n")
                print(json.dumps({key: summary[key]}), flush=True)
                if ok or (error or "").startswith("rejected"):
                    break
                time.sleep(60)
    failed = [k for k, v in summary.items() if v["status"] != "completed"]
    print(json.dumps({"done": True, "failed": failed, "total": len(summary)}), flush=True)


if __name__ == "__main__":
    main()
