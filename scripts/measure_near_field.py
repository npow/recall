"""Measure realized camera motion for fixed-action runs (forward / backward / strafe / turn / round trip).

Input: downloaded result directories named nf-<SCENE>-<path>-<time_ns> (chunk-*.mp4 + run.json).
Per chunk (including the previous chunk's last frame so the full second is covered) we fit a similarity
transform to SIFT matches between the first and last frame (RANSAC, partial affine) and report:

  scale      >1 the scene expands (camera moved toward it), <1 it contracts (camera moved away)
  tx         horizontal translation of the scene in pixels at 832 px width (A strafe -> scene moves right, +)
  dx_spread  std of per-match horizontal displacement / median |dx| : parallax proxy (near objects move more)
  yaw        essential-matrix yaw from worldsync.motion_feedback (+ = turn right)

Monocular odometry on generated video is noisy and scale-free; use it to compare actions and scenes, not
as metric motion. `usable` is False where too few matches were found.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldsync.motion_feedback import chunk_motion  # noqa: E402

WIDTH = 416


def read_chunk(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return np.stack(frames)


def match(a, b):
    def prep(x):
        h, w = x.shape[:2]
        if w > WIDTH:
            x = cv2.resize(x, (WIDTH, int(round(h * WIDTH / w))))
        return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), 832 / x.shape[1]
    ga, factor = prep(a)
    gb, _ = prep(b)
    sift = cv2.SIFT_create(nfeatures=2500)
    ka, da = sift.detectAndCompute(ga, None)
    kb, db = sift.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return None
    good = [p[0] for p in cv2.BFMatcher().knnMatch(da, db, k=2) if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 20:
        return None
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    return pa, pb, factor


def similarity(a, b):
    found = match(a, b)
    if found is None:
        return {"usable": False}
    pa, pb, factor = found
    matrix, inliers = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if matrix is None or inliers.sum() < 20:
        return {"usable": False}
    keep = inliers.ravel().astype(bool)
    dx = (pb - pa)[keep, 0]
    return {"usable": True, "scale": float(np.hypot(matrix[0, 0], matrix[1, 0])),
            "tx": float(matrix[0, 2] * factor), "inliers": int(keep.sum()),
            "dx_spread": float(np.std(dx) / (np.median(np.abs(dx)) + 1e-6))}


def summarize(run_dir, intrinsics):
    report = json.loads((run_dir / "run.json").read_text())
    fixed = report.get("fixed_actions") or []
    chunks = sorted(run_dir.glob("chunk-*.mp4"))
    previous_last, rows = None, []
    for index, path in enumerate(chunks):
        frames = read_chunk(path)
        reference = previous_last if previous_last is not None else frames[0]
        row = {"chunk": index, "action": fixed[index] if index < len(fixed) else None}
        row.update(similarity(reference, frames[-1]))
        motion = chunk_motion(frames, intrinsics, previous_last)
        row["yaw_degrees"] = motion.get("yaw_degrees")
        row["flow_px_per_s"] = motion.get("image_motion_px_per_second")
        previous_last = frames[-1]
        rows.append(row)
    by_action = {}
    for action in sorted({r["action"] for r in rows if r["action"] and r["action"] != "WAIT"}):
        sel = [r for r in rows if r["action"] == action]
        ok = [r for r in sel if r.get("usable")]
        entry = {"chunks": len(sel), "usable": len(ok)}
        if ok:
            entry["scale_cumulative"] = float(np.prod([r["scale"] for r in ok]))
            entry["scale_mean_per_chunk"] = float(np.mean([r["scale"] for r in ok]))
            entry["tx_sum_px"] = float(np.sum([r["tx"] for r in ok]))
            entry["dx_spread_mean"] = float(np.mean([r["dx_spread"] for r in ok]))
        yaws = [r["yaw_degrees"] for r in sel if r["yaw_degrees"] is not None]
        if yaws:
            entry["yaw_sum_degrees"] = float(np.sum(yaws))
            entry["yaw_chunks_measured"] = len(yaws)
        flows = [r["flow_px_per_s"] for r in sel if r["flow_px_per_s"] is not None]
        if flows:
            entry["flow_px_per_s_mean"] = float(np.mean(flows))
        by_action[action] = entry
    return {"run": run_dir.name, "task_id": report.get("task_id"), "status": report.get("status"),
            "peak_allocated_bytes": report.get("peak_allocated_bytes"), "by_action": by_action, "chunks": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    intrinsics = np.load(Path(__file__).resolve().parents[1] / "external/LingBot-World2/examples/03/intrinsics.npy")[0]
    summaries = []
    for run in args.runs:
        if (run / "run.json").exists():
            summaries.append(summarize(run, intrinsics))
    args.output.write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"{len(summaries)} runs -> {args.output}")


if __name__ == "__main__":
    main()
