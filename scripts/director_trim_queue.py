"""Drop the long insight jobs from the director dev queue (user directive: wind Modal down within ~10 minutes)."""
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.director_dev import QUEUE  # noqa: E402


def main():
    queue = modal.Queue.from_name(QUEUE, create_if_missing=True)
    kept, dropped = [], 0
    while True:
        job = queue.get(block=False)
        if job is None:
            break
        if job["split"] == "insight":
            dropped += 1
        else:
            kept.append(job)
    if kept:
        queue.put_many(kept)
    print(f"kept {len(kept)} short jobs, dropped {dropped} insight jobs")


if __name__ == "__main__":
    main()
