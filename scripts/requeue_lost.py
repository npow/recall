"""Re-queue jobs that were lost (popped by a worker that was cancelled) for one arm, ahead of the insight tail.

    MODAL_PROFILE=nissan-pow python scripts/requeue_lost.py --label full1 --arm stock gc:GC022 gc:GC023 if:IF004

Also removes stale partial attempt directories of those jobs from the results volume so the retry starts clean.
Used once: see reproduction/full-run-log.md (23:35 entry).
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="full1")
    parser.add_argument("--arm", required=True)
    parser.add_argument("cases", nargs="+", help="split:TASK")
    args = parser.parse_args()
    import modal
    from worldsync import lingbot_episode_queue as q
    queue = modal.Queue.from_name(f"{args.label}-{args.arm}")
    items = list(queue.iterate())
    present = {job["id"] for job in items}
    extra = []
    for item in args.cases:
        split, task = item.split(":")
        for attempt in ("a0", "a1", "a2"):
            subprocess.run(["modal", "volume", "rm", "-r", "worldsync-lingbot-results",
                            f"adaptive/{args.label}/{args.arm}/{split}-{task}-seed42__{attempt}"], capture_output=True)
        if q.case_key(args.arm, split, task, 42) not in present:
            extra.append(q.episode_job(args.label, args.arm, split, task, 42))
    new = [j for j in items if j["split"] != "insight"] + extra + [j for j in items if j["split"] == "insight"]
    queue.clear()
    for start in range(0, len(new), 100):
        queue.put_many(new[start:start + 100])
    print(f"{args.arm} queue {len(items)} -> {len(new)}; re-queued {[j['id'] for j in extra]}")


if __name__ == "__main__":
    main()
