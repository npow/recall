"""Visual-odometry audit: does a generated LingBot video realize its commanded camera path?

For frame pairs (i, i+STEP) we match SIFT features, fit an essential matrix with the
native intrinsics and recover relative rotation + translation direction. Yaw is read from
R about the camera y axis. Translation scale is unobservable monocularly, so we also report
the median feature flow magnitude after de-rotation (parallax proxy) and the t_z sign.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

STEP = 4


def commanded_yaw(R):
    return float(np.degrees(np.arctan2(R[0, 2], R[2, 2])))


def load_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    return frames


def pair_motion(a, b, K, sift, matcher):
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return None
    good = [m for m, n in matcher.knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
    if len(good) < 20:
        return None
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    E, mask = cv2.findEssentialMat(pa, pb, K, cv2.RANSAC, 0.999, 1.0)
    if E is None or E.shape != (3, 3):
        return None
    inl, R, t, mask2 = cv2.recoverPose(E, pa, pb, K, mask=mask)
    # R maps camera-a coordinates to camera-b coordinates; camera motion is its inverse.
    yaw = -commanded_yaw(R)
    # De-rotated flow: remove the pure-rotation homography K R K^-1 from pb.
    H = K @ R @ np.linalg.inv(K)
    warped = cv2.perspectiveTransform(pb.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
    flow = np.linalg.norm(warped - pa, axis=1)[mask.ravel() > 0]
    return {"yaw_deg": yaw, "inliers": int(inl), "matches": len(good),
            "t_dir": (-R.T @ t).ravel().tolist(), "derot_flow_px": float(np.median(flow))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="directory with rollout.mp4 and controls/")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    frames = load_frames(args.run / "rollout.mp4")
    Ks = np.load(args.run / "controls/intrinsics.npy")
    poses = np.load(args.run / "controls/poses.npy")
    fx, fy, cx, cy = Ks[0]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    sift = cv2.SIFT_create(2000)
    matcher = cv2.BFMatcher()
    rows = []
    n = min(len(frames), len(poses))
    for i in range(0, n - STEP, STEP):
        m = pair_motion(frames[i], frames[i + STEP], K, sift, matcher)
        Ra, Rb = poses[i][:3, :3], poses[i + STEP][:3, :3]
        cmd_yaw = commanded_yaw(Ra.T @ Rb)
        cmd_t = poses[i + STEP][:3, 3] - poses[i][:3, 3]
        rows.append({"frame": i, "cmd_yaw_deg": cmd_yaw,
                     "cmd_translation_norm": float(np.linalg.norm(cmd_t)),
                     "real": m})
    yaw_cmd = np.cumsum([r["cmd_yaw_deg"] for r in rows])
    yaw_real = np.cumsum([(r["real"] or {}).get("yaw_deg", 0.0) for r in rows])
    for r, c, v in zip(rows, yaw_cmd, yaw_real):
        r["cum_cmd_yaw"], r["cum_real_yaw"] = float(c), float(v)
    args.out.write_text(json.dumps(rows, indent=1))
    print("frame cmd_yaw real_yaw cum_cmd cum_real flow_px inliers cmd_t")
    for r in rows:
        m = r["real"] or {}
        print(f'{r["frame"]:4d} {r["cmd_yaw_deg"]:7.2f} {m.get("yaw_deg", float("nan")):7.2f} '
              f'{r["cum_cmd_yaw"]:8.1f} {r["cum_real_yaw"]:8.1f} {m.get("derot_flow_px", float("nan")):6.2f} '
              f'{m.get("inliers", 0):4d} {r["cmd_translation_norm"]:.2f}')


if __name__ == "__main__":
    main()
