"""Submit the 'protocol-parity bug fix v2' regeneration: 7 tasks x 2 arms, seed 42, frozen flags, <= 3 containers.

    MODAL_PROFILE=nissan-pow ADAPTIVE_MAX_CONTAINERS=3 modal deploy scripts/modal_lingbot_adaptive_runner.py
    MODAL_PROFILE=nissan-pow python scripts/submit_fix2.py --label fix2 --stock-workers 1 --budget-workers 2

All workers run on the H200 class (the frozen run's stock episodes mostly ran there too). Results land under
adaptive/<label>/<arm>/... in the results volume; scoring uses scripts/score_pipeline.py --label <label> --no-refresh.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASES = [("gc", "GC008"), ("gc", "GC016"), ("outsight", "OE028"), ("outsight", "OE030"),
         ("outsight", "OE032"), ("outsight", "OE036"), ("outsight", "OE047")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="fix2")
    parser.add_argument("--stock-workers", type=int, default=1)
    parser.add_argument("--budget-workers", type=int, default=2)
    args = parser.parse_args()
    import modal
    from worldsync import lingbot_episode_queue as q
    if args.stock_workers + args.budget_workers > 3:
        raise SystemExit("credit cap: at most 3 containers")
    out = ROOT / "outputs/adaptive" / args.label
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps({"label": args.label, "arms": ["stock", "budget"], "seeds": [42],
        "cases": [f"{s}:{t}" for s, t in CASES], "containers": {"stock": args.stock_workers, "budget": args.budget_workers},
        "note": "protocol-parity bug fix v2 (u/j pitch keys in plan parsing); frozen flags unchanged"}, indent=1) + "\n")
    cls = modal.Cls.from_name("worldsync-lingbot-adaptive-runner", "AdaptiveH200")()
    calls = {}
    for arm, workers in (("stock", args.stock_workers), ("budget", args.budget_workers)):
        queue = modal.Queue.from_name(f"{args.label}-{arm}", create_if_missing=True)
        queue.clear()
        queue.put_many([q.episode_job(args.label, arm, split, task, 42) for split, task in CASES])
        spawned = [cls.drain.spawn(f"{args.label}-{arm}", args.label, None, 3, False) for _ in range(workers)]
        calls[arm] = [call.object_id for call in spawned]
        print(f"{arm}: queued {len(CASES)} episodes, spawned {workers} workers on H200", flush=True)
    (out / "calls.json").write_text(json.dumps(calls, indent=1) + "\n")


if __name__ == "__main__":
    main()
