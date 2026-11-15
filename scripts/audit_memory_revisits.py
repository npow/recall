"""Was the revisited view still in the bounded archive when the camera came back? (read-only)

For every budget episode of the full run, rebuild commanded latent poses from the
executed actions, find 'revisit chunks' (the chunk's target pose is within COST of
some latent frame older than the stock 12-frame recent window and outside the sink),
and check, from the recorded memory trace, whether that old frame was (a) still in
the archive and (b) actually retrieved. Also reports the archive's oldest frame per
chunk to show whether the bounded archive behaves as a FIFO.

Usage: python scripts/audit_memory_revisits.py EPISODES_DIR [--json OUT]
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldsync.lingbot_budgetmem import pose_cost  # noqa: E402
from worldsync.lingbot_controls import CameraPlan  # noqa: E402

COST = 0.5  # 2 normalized units or 30 degrees
STOCK_RECENT = 12
SINK = 6


def main():
    root = Path(sys.argv[1])
    out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    totals = collections.defaultdict(lambda: collections.Counter())
    rows = []
    for directory in sorted(root.glob("budget-*-seed42")):
        split = directory.name.split("-")[1]
        run = json.loads((directory / "run.json").read_text())
        trace = {row["chunk"]: row for row in (run.get("memory") or {}).get("trace", [])}
        actions = [a["action"] for a in run["actions"] if a.get("operation") != "END" and "pose_after" in a]
        camera = CameraPlan()
        ends = []
        for action in actions:
            camera.advance(action)
            ends.append(len(camera.poses))
        poses = np.stack(camera.poses) if camera.poses else np.zeros((0, 4, 4))
        counter = totals[split]
        counter["episodes"] += 1
        episode = {"episode": directory.name, "chunks": len(ends), "revisit_chunks": 0, "old_available": 0,
                   "old_retrieved": 0, "original_evicted": 0, "sink_revisit_chunks": 0, "fifo_chunks": 0, "archive_full_chunks": 0}
        for chunk, end in enumerate(ends):
            first = chunk * 4
            target = poses[end - 1]
            row = trace.get(chunk)
            if row is None:
                continue
            archive = set(row["archive_frame_ids"])
            if len(archive) >= 24:
                episode["archive_full_chunks"] += 1
                # FIFO signature: archive is exactly the newest frames before the protected tail
                if min(archive) >= first - 4 - 24 - 4:
                    episode["fifo_chunks"] += 1
            old = [i for i in range(SINK, max(first - STOCK_RECENT, SINK)) if pose_cost(target, poses[i]) <= COST]
            sink_match = any(pose_cost(target, poses[i]) <= COST for i in range(0, min(SINK, first - STOCK_RECENT)))
            if sink_match and first >= STOCK_RECENT + SINK:
                episode["sink_revisit_chunks"] += 1
            if not old:
                continue
            episode["revisit_chunks"] += 1
            if any(i in archive for i in old):
                episode["old_available"] += 1
            if any(i in row["retrieved_frame_ids"] for i in old) and row["attention_replaced"]:
                episode["old_retrieved"] += 1
            if min(old) not in archive:
                episode["original_evicted"] += 1
        rows.append(episode)
        for key in ("chunks", "revisit_chunks", "old_available", "old_retrieved", "original_evicted",
                    "sink_revisit_chunks", "fifo_chunks", "archive_full_chunks"):
            counter[key] += episode[key]
        counter["episodes_with_revisit"] += episode["revisit_chunks"] > 0
        counter["episodes_revisit_retrieved"] += episode["old_retrieved"] > 0
    print("split episodes ep_with_revisit ep_revisit_retrieved chunks revisit_chunks old_available old_retrieved original_evicted sink_revisit archive_full fifo")
    for split, c in sorted(totals.items()):
        print(split, c["episodes"], c["episodes_with_revisit"], c["episodes_revisit_retrieved"], c["chunks"], c["revisit_chunks"],
              c["old_available"], c["old_retrieved"], c["original_evicted"], c["sink_revisit_chunks"], c["archive_full_chunks"], c["fifo_chunks"])
    if out:
        Path(out).write_text(json.dumps({"cost_threshold": COST, "by_split": {k: dict(v) for k, v in totals.items()},
                                         "episodes": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
