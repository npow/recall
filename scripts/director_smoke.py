"""Print director texts for a few tasks (LLM calls only; no GPU, no judge). Usage: python scripts/director_smoke.py"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worldsync.credentials import load_local_credentials  # noqa: E402
from worldsync.lingbot_director import Director  # noqa: E402


def task(split, task_id):
    return next(r for r in json.loads((ROOT / f"external/PlayWorld/data/{split}/data.json").read_text()) if r["task_id"] == task_id)


def main():
    load_local_credentials()
    from anthropic import Anthropic
    client = Anthropic(timeout=60, max_retries=2)
    plans = [("outsight", "OE014", "unseen", [(1, "LEFT", -60, 0.8), (2, "LEFT", -90, 1.8), (3, "RIGHT", -30, 2.8), (4, "RIGHT", 0, 3.8)]),
             ("insight", "OE001", "perchunk", [(1, "WAIT", 0, 0.8), (5, "WAIT", 0, 4.8)]),
             ("if", "IF001", "perchunk", [(1, "W", 0, 0.8)])]
    for split, task_id, mode, steps in plans:
        record = task(split, task_id)
        director = Director(client, record, mode, interval=4)
        print("==", task_id, mode, "| objective:", record["prompt"][:100])
        print(" INIT:", director.initial())
        for index, action, heading, elapsed in steps:
            text, changed = director.for_chunk(index, action, {"heading_degrees": heading, "position_normalized_units": [0, 0, 0]}, elapsed)
            print(f" c{index} {action} changed={changed}:", text[:420])
            print("    state:", director.world_state[:200])
        print(" errors", director.metadata()["errors"])


if __name__ == "__main__":
    main()
