"""Run visual odometry on several round-trip runs and summarize realized vs commanded motion.

Reports, per run: realized yaw summed over each commanded turn block (reliable estimates only:
>= MIN_INLIERS inliers), approach scale (similarity-transform scale between frames 16 apart)
per forward block, and DINO-free start/end frame similarity left to other scripts.
usage: compare_realized_motion.py OUT.json LABEL=RUN_DIR [LABEL=RUN_DIR ...]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from realized_motion import STEP, commanded_yaw, load_frames, pair_motion  # noqa: E402

MIN_INLIERS = 30
GAP = 16


def blocks(poses):
    """Commanded turn/forward blocks from frame-wise commanded poses."""
    n = len(poses)
    yaw = [commanded_yaw(poses[i][:3, :3].T @ poses[i + STEP][:3, :3]) for i in range(0, n - STEP, STEP)]
    move = [float(np.linalg.norm(poses[i + STEP][:3, 3] - poses[i][:3, 3])) for i in range(0, n - STEP, STEP)]
    kind = ["turn" if abs(y) > 1 else ("move" if m > 1e-6 else "hold") for y, m in zip(yaw, move)]
    out, start = [], 0
    for i in range(1, len(kind) + 1):
        if i == len(kind) or kind[i] != kind[start]:
            out.append((kind[start], start * STEP, i * STEP))
            start = i
    return out


def scale_between(a, b, sift, matcher):
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None:
        return None
    good = [m for m, n in matcher.knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
    if len(good) < 15:
        return None
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None or int(inl.sum()) < 15:
        return None
    return float(np.hypot(M[0, 0], M[1, 0]))


def analyze(run):
    run = Path(run)
    frames = load_frames(run / "rollout.mp4")
    Ks = np.load(run / "controls/intrinsics.npy")
    poses = np.load(run / "controls/poses.npy")
    fx, fy, cx, cy = Ks[0]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    sift, matcher = cv2.SIFT_create(2000), cv2.BFMatcher()
    n = min(len(frames), len(poses))
    result = []
    for kind, lo, hi in blocks(poses[:n]):
        row = {"kind": kind, "frames": [lo, hi]}
        if kind == "turn":
            cmd, real, used = 0.0, 0.0, 0
            for i in range(lo, min(hi, n - STEP), STEP):
                cmd += commanded_yaw(poses[i][:3, :3].T @ poses[i + STEP][:3, :3])
                m = pair_motion(frames[i], frames[i + STEP], K, sift, matcher)
                if m and m["inliers"] >= MIN_INLIERS and abs(m["yaw_deg"]) < 40:
                    real += m["yaw_deg"]
                    used += 1
            steps = len(range(lo, min(hi, n - STEP), STEP))
            row.update(commanded_yaw_deg=round(cmd, 1), realized_yaw_reliable_deg=round(real, 1),
                       reliable_steps=f"{used}/{steps}")
            # Scale the reliable sum to all steps for a rough whole-block estimate.
            row["realized_yaw_extrapolated_deg"] = round(real * steps / used, 1) if used else None
        elif kind == "move":
            scales = []
            for i in range(lo, min(hi, n) - GAP, GAP):
                s = scale_between(frames[i], frames[i + GAP], sift, matcher)
                if s:
                    scales.append(s)
            row["approach_scale_per_16f"] = [round(s, 3) for s in scales]
            row["approach_scale_total"] = round(float(np.prod(scales)), 3) if scales else None
        result.append(row)
    return result


def main():
    out = Path(sys.argv[1])
    runs = dict(a.split("=", 1) for a in sys.argv[2:])
    data = {label: analyze(path) for label, path in runs.items()}
    out.write_text(json.dumps(data, indent=1))
    for label, rows in data.items():
        print("==", label)
        for r in rows:
            if r["kind"] == "hold":
                continue
            print(" ", r)


if __name__ == "__main__":
    main()
