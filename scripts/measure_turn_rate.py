"""Estimate realized camera yaw from generated frames (rotation-only homography).

For each chunk video it chains frame-to-frame homographies, converts them to
rotations with the native intrinsics, and reports realized yaw per chunk next
to the commanded action. Static-scene feature matching only; moving subjects
are rejected by RANSAC. Read-only analysis of existing rollout directories.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STRIDE = 3


def intrinsics(width, height):
    fx, fy, cx, cy = np.load(ROOT / "external/LingBot-World2/examples/03/intrinsics.npy")[0]
    sx, sy = width / 832, height / 480
    return np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]])


def pair_yaw(a, b, K, detector, matcher):
    ga, gb = (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (a, b))
    ka, da = detector.detectAndCompute(ga, None)
    kb, db = detector.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return None
    good = [m for m, n in matcher.knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
    if len(good) < 25:
        return None
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    if H is None or mask.sum() < 20:
        return None
    R = np.linalg.inv(K) @ H @ K
    R /= np.cbrt(np.linalg.det(R))
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    # Camera yaw about the vertical axis; positive = scene content moves left (turn right).
    return math.degrees(math.atan2(-R[0, 2], R[2, 2])), int(mask.sum())


def measure(directory, actions):
    detector, matcher = cv2.SIFT_create(2000), cv2.BFMatcher()
    rows, total = [], 0.0
    for index, movie in enumerate(sorted(Path(directory).glob("chunk-*.mp4"))):
        frames = [np.asarray(f) for f in imageio.get_reader(movie)]
        K = intrinsics(frames[0].shape[1], frames[0].shape[0])
        chunk, inliers, failed = 0.0, [], 0
        for i in range(0, len(frames) - STRIDE, STRIDE):
            result = pair_yaw(frames[i], frames[i + STRIDE], K, detector, matcher)
            if result is None:
                failed += 1
                continue
            chunk += result[0]
            inliers.append(result[1])
        total += chunk
        seconds = (len(frames) - 1) / 16
        rows.append({"chunk": index, "action": actions[index] if index < len(actions) else None, "frames": len(frames),
            "realized_yaw_deg": round(chunk, 1), "realized_deg_per_s": round(chunk / seconds, 1),
            "median_inliers": int(np.median(inliers)) if inliers else 0, "failed_pairs": failed})
    return rows, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {}
    for directory in args.directories:
        run = json.loads((directory / "run.json").read_text())
        actions = [a["action"] for a in run["actions"]]
        rows, total = measure(directory, actions)
        report[directory.name] = {"task_id": run["task_id"], "rows": rows, "net_realized_yaw_deg": round(total, 1),
            "commanded_net_yaw_deg": run.get("camera_diagnostics", {}).get("net_yaw_degrees")}
        print(directory.name)
        for row in rows:
            print(" ", row)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
