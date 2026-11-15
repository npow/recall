"""Audit the budget-memory trace of full-run episodes (read-only).

Usage: python scripts/audit_memory_trace.py EPISODES_DIR [case ...]
Prints, per budget episode: chunks, actions, and for each chunk the retrieved
frame ids with their pose cost to the chunk's target pose (recomputed from the
commanded actions with CameraPlan), so we can check that retrieval picks the
revisited view from the same episode.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldsync.lingbot_budgetmem import pose_cost  # noqa: E402
from worldsync.lingbot_controls import CameraPlan  # noqa: E402


def poses_for(actions):
    camera = CameraPlan()
    per_chunk = []
    for action in actions:
        camera.advance(action)
        per_chunk.append(len(camera.poses))
    return np.stack(camera.poses), per_chunk


def main():
    root = Path(sys.argv[1])
    cases = sys.argv[2:]
    for name in cases:
        run = json.loads((root / f"budget-{name}-seed42" / "run.json").read_text())
        actions = [a["action"] for a in run["actions"] if a.get("operation") != "END" and "pose_after" in a]
        poses, counts = poses_for(actions)
        memory = run.get("memory") or {}
        trace = memory.get("trace", [])
        print(f"\n== {name}: chunks={len(run['chunks'])} actions={''.join(a[0] if a in 'WASD' else a[0].lower() for a in actions)}"
              f" rope={memory.get('rope')} calls={memory.get('retrieval_attention_calls')} archive_GB={memory.get('archive_bytes', 0) / 1e9:.1f}")
        for row in trace:
            chunk = row["chunk"]
            if chunk >= len(counts):
                continue
            target = poses[counts[chunk] - 1]
            ids = row["retrieved_frame_ids"]
            if not ids:
                continue
            costs = [pose_cost(target, poses[i]) for i in ids if i < len(poses)]
            archive = row["archive_frame_ids"]
            best_all = min((pose_cost(target, poses[i]), i) for i in archive if i < len(poses)) if archive else None
            print(f" chunk {chunk:2d} first_frame {chunk * 4:3d} replaced={row['attention_replaced']} ids={ids}"
                  f" cost[min/max]={min(costs):.2f}/{max(costs):.2f} archive={len(archive)} (max id {max(archive)})"
                  f" best_in_archive={best_all[0]:.2f}@{best_all[1]}")


if __name__ == "__main__":
    main()
