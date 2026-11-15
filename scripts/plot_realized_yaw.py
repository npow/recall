"""Plot commanded vs realized cumulative yaw for one run from realized_motion.py output."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = json.load(open(sys.argv[1]))
frames, cmd, real = [], [], []
c = r = 0.0
for row in rows:
    c += row["cmd_yaw_deg"]
    m = row["real"]
    if m and m["inliers"] >= 30 and abs(m["yaw_deg"]) < 40:
        r += m["yaw_deg"]          # reliable estimate
    else:
        r += row["cmd_yaw_deg"] * 0  # unreliable: held flat, not counted as motion
    frames.append(row["frame"]); cmd.append(c); real.append(r)
plt.figure(figsize=(8, 3.6))
plt.plot(frames, cmd, label="commanded yaw (deg)")
plt.plot(frames, real, label="realized yaw, reliable VO steps only")
plt.xlabel("frame (16 FPS)"); plt.ylabel("cumulative yaw (deg)")
plt.title("Stock round_trip GC007: second turn mostly unmeasurable (few inliers)")
plt.legend(); plt.tight_layout(); plt.savefig(sys.argv[2], dpi=140)
