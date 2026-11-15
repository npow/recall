"""Re-order the full-run queues (insight last) and optionally move stock workers to H200.

    MODAL_PROFILE=nissan-pow python scripts/requeue_full_run.py --label full1 --stock-gpu H200

Used once during the full run (see reproduction/full-run-log.md): the coordinator asked for the heavy
60-chunk insight episodes to go last while an insight-length diagnosis is in flight, and the H100 stock
workers never got a GPU. Episodes already popped by a worker are not re-queued; finished episodes are
skipped by the workers anyway (resume).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPLITS = ("gc", "if", "outsight")  # interleaved first; insight appended last


def ordered_cases(data_root):
    lanes = {}
    for split in (*SPLITS, "insight"):
        records = json.loads((Path(data_root) / split / "data.json").read_text())
        lanes[split] = sorted((record["task_id"] for record in records),
                              key=lambda task: hashlib.sha256(f"worldsync-fullrun-v1:{task}".encode()).hexdigest())
    ordered, index = [], 0
    while any(index < len(lanes[s]) for s in SPLITS):
        for split in SPLITS:
            if index < len(lanes[split]):
                ordered.append((split, lanes[split][index]))
        index += 1
    return ordered + [("insight", task) for task in lanes["insight"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="full1")
    parser.add_argument("--stock-gpu", choices=("H100", "H200"), default="H200")
    parser.add_argument("--stock-workers", type=int, default=5)
    parser.add_argument("--data-root", type=Path, default=ROOT / "external/PlayWorld/data")
    parser.add_argument("--skip-stock-respawn", action="store_true")
    args = parser.parse_args()
    import modal
    from worldsync import lingbot_episode_queue as q
    cases = ordered_cases(args.data_root)
    out = ROOT / "outputs/adaptive" / args.label
    calls = json.loads((out / "calls.json").read_text())
    seed = 42
    for arm in ("stock", "budget"):
        queue = modal.Queue.from_name(f"{args.label}-{arm}", create_if_missing=True)
        remaining = queue.len()
        popped = len(cases) - remaining  # the first `popped` jobs of the ORIGINAL order are already taken
        original = json.loads((out / "plan.json").read_text())["cases"]
        taken = {tuple(item.split(":")) for item in original[:popped]}
        jobs = [q.episode_job(args.label, arm, split, task, seed) for split, task in cases if (split, task) not in taken]
        queue.clear()
        for start in range(0, len(jobs), 100):
            queue.put_many(jobs[start:start + 100])
        print(f"{arm}: {popped} jobs already taken, re-queued {len(jobs)} (insight last)", flush=True)
    if not args.skip_stock_respawn and args.stock_gpu == "H200":
        for object_id in calls.get("stock", []):
            try:
                modal.FunctionCall.from_id(object_id).cancel()
            except Exception as error:  # already finished or gone
                print("cancel", object_id, type(error).__name__)
        cls = modal.Cls.from_name("worldsync-lingbot-adaptive-runner", "AdaptiveH200")()
        spawned = [cls.drain.spawn(f"{args.label}-stock", args.label, None, 3, False) for _ in range(args.stock_workers)]
        calls["stock_h200"] = [call.object_id for call in spawned]
        (out / "calls.json").write_text(json.dumps(calls, indent=1) + "\n")
        print(f"stock: cancelled H100 workers, spawned {len(spawned)} workers on H200", flush=True)


if __name__ == "__main__":
    main()
