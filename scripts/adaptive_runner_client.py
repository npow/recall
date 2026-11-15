"""Client for the warm adaptive-episode runner: submit episodes, spawn drain workers, read status.

    MODAL_PROFILE=nissan-pow python scripts/adaptive_runner_client.py submit --label full1 \
        --arms stock,budget --all --seeds 42 --containers stock=5,budget=5
    MODAL_PROFILE=nissan-pow python scripts/adaptive_runner_client.py status --label full1

Submitting again is safe: finished episodes are skipped (resume). Needs the app deployed first:
    MODAL_PROFILE=nissan-pow modal deploy scripts/modal_lingbot_adaptive_runner.py
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
APP_NAME = "worldsync-lingbot-adaptive-runner"
SPLITS = ("gc", "if", "insight", "outsight")
CLASS_FOR_GPU = {"H200": "AdaptiveH200", "H100": "AdaptiveH100"}


def all_cases(data_root):
    cases = []
    for split in SPLITS:
        for record in json.loads((Path(data_root) / split / "data.json").read_text()):
            cases.append((split, record["task_id"]))
    return cases


def select_cases(data_root, per_split=None, only=None, salt="worldsync-final"):
    """All cases, or `per_split` cases per split ordered by a salted hash of the case id (never by outcome)."""
    cases = all_cases(data_root)
    if only:
        wanted = {tuple(item.split(":")) for item in only}
        return [case for case in cases if case in wanted]
    if per_split is None:
        return cases
    chosen = []
    for split in SPLITS:
        rows = sorted((c for c in cases if c[0] == split),
                      key=lambda c: hashlib.sha256(f"{salt}:{c[1]}".encode()).hexdigest())
        chosen += rows[:per_split]
    return chosen


def parse_containers(text):
    return {key: int(value) for key, value in (item.split("=") for item in text.split(","))}


def submit(args):
    import modal
    from worldsync import lingbot_episode_queue as q
    cases = select_cases(args.data_root, args.per_split if not args.all else None, args.cases.split(",") if args.cases else None)
    arms = args.arms.split(",")
    containers = parse_containers(args.containers)
    seeds = [int(s) for s in args.seeds.split(",")]
    plan = {"label": args.label, "arms": arms, "seeds": seeds, "cases": [f"{s}:{t}" for s, t in cases],
            "containers": containers, "max_attempts": args.max_attempts, "canary": args.canary}
    out = ROOT / "outputs/adaptive" / args.label
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    calls = {}
    for arm in arms:
        jobs = [q.episode_job(args.label, arm, split, task, seed) for seed in seeds for split, task in cases]
        queue = modal.Queue.from_name(f"{args.label}-{arm}", create_if_missing=True)
        queue.clear()
        for start in range(0, len(jobs), 100):
            queue.put_many(jobs[start:start + 100])
        cls = modal.Cls.from_name(APP_NAME, CLASS_FOR_GPU[q.ARMS[arm]["gpu"]])()
        spawned = [cls.drain.spawn(f"{args.label}-{arm}", args.label, None, args.max_attempts, args.canary)
                   for _ in range(containers.get(arm, 1))]
        calls[arm] = [call.object_id for call in spawned]
        print(f"{arm}: queued {len(jobs)} episodes, spawned {len(spawned)} drain workers", flush=True)
    (out / "calls.json").write_text(json.dumps(calls, indent=1) + "\n")


def status(args):
    import modal
    function = modal.Function.from_name(APP_NAME, "status")
    report = function.remote(args.label)
    for arm, rows in report["arms"].items():
        done = rows["completed_count"]
        print(f"{arm}: completed {done}, failed {len(rows['failed'])}, skipped {rows['skipped']}")
        if rows["throughput"]:
            t = rows["throughput"]
            print(f"   mean episode {t['mean_episode_seconds']:.0f} s, {t['episodes_per_hour_per_container']:.1f}/h/container")
    out = ROOT / "outputs/adaptive" / args.label
    out.mkdir(parents=True, exist_ok=True)
    (out / "status.json").write_text(json.dumps(report, indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--label", required=True)
    s.add_argument("--arms", default="stock,budget")
    s.add_argument("--all", action="store_true", help="all 171 cases")
    s.add_argument("--per-split", type=int, help="hash-ordered subset, N per split")
    s.add_argument("--cases", help="explicit list split:TASK,split:TASK")
    s.add_argument("--seeds", default="42")
    s.add_argument("--containers", default="stock=5,budget=5")
    s.add_argument("--max-attempts", type=int, default=3)
    s.add_argument("--canary", action="store_true", help="run a fixed-path canary before/after to verify no state leakage")
    s.add_argument("--data-root", type=Path, default=ROOT / "external/PlayWorld/data")
    s.set_defaults(func=submit)
    t = sub.add_parser("status")
    t.add_argument("--label", required=True)
    t.set_defaults(func=status)
    args = parser.parse_args()
    if args.command == "submit" and not (args.all or args.per_split or args.cases):
        parser.error("choose --all, --per-split N or --cases")
    args.func(args)


if __name__ == "__main__":
    main()
