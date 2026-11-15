"""Find loop-closure / revisit pairs in a camera trajectory (OpenCV c2w poses).

A revisit is a pair of frames (i, j), j - i >= min_gap frames, where the camera
at j is close to and looks the same way as at i (high view overlap proxy), but
between them the camera left that view (overlap dropped low). Overlap is a pose
only proxy: distance in units of the trajectory length and heading difference.
It does not check scene content, so it is a candidate generator, not ground truth.
"""
import numpy as np

FPS = 16


def overlap(poses, i, ks, radius, max_angle_deg=45.0):
    """Proxy view overlap in [0,1] between frame i and frames `ks`."""
    p = poses[:, :3, 3]
    f = poses[:, :3, 2]  # camera z axis (forward in OpenCV) in world coordinates
    d = np.linalg.norm(p[ks] - p[i], axis=-1)
    cos = np.clip(f[ks] @ f[i], -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    pos_term = np.exp(-((d / radius) ** 2))
    ang_term = np.clip(1.0 - ang / max_angle_deg, 0.0, 1.0)
    return pos_term * ang_term


def find_revisits(poses, min_gap_s=8.0, max_gap_s=24.0, key_stride=8, radius_frac=0.06,
                  high=0.55, low=0.12, max_pairs=3, min_key_separation_s=8.0, fps=FPS):
    """Return up to `max_pairs` revisit dicts for one trajectory [N,4,4]."""
    poses = np.asarray(poses, dtype=np.float64)
    n = len(poses)
    p = poses[:, :3, 3]
    path = float(np.linalg.norm(np.diff(p, axis=0), axis=-1).sum())
    if path < 1e-6:
        return []
    radius = radius_frac * path
    min_gap, max_gap = int(min_gap_s * fps), int(max_gap_s * fps)
    candidates = []
    for i in range(0, n - min_gap, key_stride):
        hi = min(n, i + max_gap + 1)
        js = np.arange(i + min_gap, hi, 4)  # latent-chunk grid
        if len(js) == 0:
            continue
        ov_j = overlap(poses, i, js, radius)
        ks_all = np.arange(i + 1, hi)
        ov_all = overlap(poses, i, ks_all, radius)
        for j, o in zip(js, ov_j):
            if o < high:
                continue
            between = ov_all[: j - (i + 1)]
            if len(between) == 0:
                continue
            dip = float(between.min())
            if dip > low:
                continue  # never left the view
            candidates.append({"keyframe": int(i), "revisit": int(j), "overlap": float(o),
                               "min_overlap_between": dip, "gap_s": (j - i) / fps,
                               "score": float(o * (1.0 - dip))})
    candidates.sort(key=lambda c: -c["score"])
    chosen = []
    for c in candidates:
        if all(abs(c["keyframe"] - s["keyframe"]) >= min_key_separation_s * fps for s in chosen):
            chosen.append(c)
        if len(chosen) >= max_pairs:
            break
    return chosen
