"""Print the per-scene x action table from nf-measure.json.

    python3 scripts/nf_table.py outputs/nf-measure.json [--markdown]
Columns: scene | forward (W) cum. scale | backward (S) cum. scale | A tx | D tx | turn yaw | round-trip W scale
"""
import json
from pathlib import Path
import re
import sys

runs = json.loads(Path(sys.argv[1]).read_text())
rows = {}
for r in runs:
    match = re.match(r"nf-(GC\d+)-([a-z0-9_]+?)-\d+$", r["run"])
    if not match:
        continue
    scene, path = match.groups()
    rows.setdefault(scene, {})[path] = r["by_action"]


def get(scene, path, action, key, digits=2):
    value = rows.get(scene, {}).get(path, {}).get(action, {}).get(key)
    return "" if value is None else f"{value:.{digits}f}"


header = ["scene", "W scale", "S scale", "A tx px", "D tx px", "turn yaw", "rt W scale", "W flow", "S flow"]
lines = [header]
for scene in sorted(rows):
    lines.append([scene, get(scene, "fwd8", "W", "scale_cumulative"), get(scene, "bwd8", "S", "scale_cumulative"),
                  get(scene, "left8", "A", "tx_sum_px", 0), get(scene, "right8", "D", "tx_sum_px", 0),
                  get(scene, "turn180", "RIGHT", "yaw_sum_degrees", 0), get(scene, "round_trip", "W", "scale_cumulative"),
                  get(scene, "fwd8", "W", "flow_px_per_s_mean", 1), get(scene, "bwd8", "S", "flow_px_per_s_mean", 1)])
if "--markdown" in sys.argv:
    print("| " + " | ".join(lines[0]) + " |")
    print("|" + "---|" * len(lines[0]))
    for line in lines[1:]:
        print("| " + " | ".join(line) + " |")
else:
    for line in lines:
        print("  ".join(f"{c:>9}" for c in line))
