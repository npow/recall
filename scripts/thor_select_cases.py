"""Pick the 12 seed-noise cases: first 3 per split in the full run's salted-hash order (never by outcome).

Insight development cases (used to choose the insight text prompt) are skipped so they stay out of the held-out set.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("gc", "if", "insight", "outsight")
INSIGHT_DEV = {"OE008", "OE022", "OE035", "OE040", "OE063", "OE074"}
SALT = "worldsync-fullrun-v1"


def select(data_root, per_split=3):
    chosen = []
    for split in SPLITS:
        ids = [row["task_id"] for row in json.loads((Path(data_root) / split / "data.json").read_text())]
        ids.sort(key=lambda t: hashlib.sha256(f"{SALT}:{t}".encode()).hexdigest())
        if split == "insight":
            ids = [t for t in ids if t not in INSIGHT_DEV]
        chosen += [(split, t) for t in ids[:per_split]]
    return chosen


if __name__ == "__main__":
    cases = select(ROOT / "external/PlayWorld/data")
    target = ROOT / "outputs/thor-cases.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps([{"split": s, "task_id": t} for s, t in cases], indent=1) + "\n")
    print(cases, file=sys.stderr)
