"""Closed-loop measured-motion feedback for the adaptive player (experimental protocol deviation).

After each generated chunk we estimate what the camera actually did from the decoded frames:
yaw from an essential-matrix fit with the native intrinsics, and a coarse image-motion magnitude
(median matched-feature displacement) as a forward-progress proxy. Both are monocular visual
odometry on generated video: noisy, scale-free, and rejected when evidence is weak. The player
sees them next to its commanded-state tracker. No rubric text is involved.

Sign convention matches CameraPlan: positive yaw = turning right.
"""
import math

import numpy as np

STEP = 4                # frame spacing for pair estimates (a quarter second at 16 FPS)
MIN_INLIERS = 30
MIN_INLIER_RATIO = 0.3
MIN_PAIRS_FOR_YAW = 2   # reliable pairs required before a chunk yaw is reported
WORK_WIDTH = 416        # downscale for speed; intrinsics are rescaled to match


def _scaled_intrinsics(intrinsics, width, height):
    """intrinsics = LingBot [fx, fy, cx, cy] defined for 832x480 frames, or a 3x3 matrix already in pixels."""
    values = np.asarray(intrinsics, dtype=float)
    if values.shape == (3, 3):
        return values
    fx, fy, cx, cy = values[:4] * np.array([width / 832, height / 480, width / 832, height / 480])
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def pair_motion(first_rgb, second_rgb, intrinsics):
    """Return {usable, yaw_degrees, scale, inliers} for one frame pair; usable=False when evidence is weak."""
    import cv2
    height, width = first_rgb.shape[:2]
    factor = WORK_WIDTH / width if width > WORK_WIDTH else 1.0
    if factor != 1.0:
        size = (int(round(width * factor)), int(round(height * factor)))
        first_rgb, second_rgb = cv2.resize(first_rgb, size), cv2.resize(second_rgb, size)
    K = _scaled_intrinsics(intrinsics, first_rgb.shape[1], first_rgb.shape[0])
    detector = cv2.SIFT_create(nfeatures=2000)
    ka, da = detector.detectAndCompute(cv2.cvtColor(first_rgb, cv2.COLOR_RGB2GRAY), None)
    kb, db = detector.detectAndCompute(cv2.cvtColor(second_rgb, cv2.COLOR_RGB2GRAY), None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return {"usable": False, "reason": "few features"}
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < MIN_INLIERS:
        return {"usable": False, "reason": "few matches", "matches": len(good)}
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    E, mask = cv2.findEssentialMat(pa, pb, K, cv2.RANSAC, 0.999, 1.0)
    if E is None or E.shape != (3, 3):
        return {"usable": False, "reason": "essential matrix failed", "matches": len(good)}
    inliers = int(mask.sum())
    if inliers < MIN_INLIERS or inliers / len(good) < MIN_INLIER_RATIO:
        return {"usable": False, "reason": "few inliers", "inliers": inliers}
    # decomposeEssentialMat returns the true rotation and its 180-degree flip about the translation axis;
    # within a quarter second the true one is the small rotation. recoverPose's cheirality test fails for
    # near-pure rotations, so it is not used.
    first, second, _ = cv2.decomposeEssentialMat(E)
    R = min((first, second), key=lambda r: math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2))))
    # R maps camera-a coordinates to camera-b; camera motion is the inverse rotation.
    yaw = -math.degrees(math.atan2(R[0, 2], R[2, 2]))
    return {"usable": True, "yaw_degrees": yaw, "inliers": inliers}


def image_motion(first_rgb, last_rgb, seconds):
    """Median matched-feature displacement (px per second at a 416-px frame width), or None if unmatched.

    Heuristic reference from one stock GC007 video: holding still is about 1 px/s; stepping toward a distant
    landmark is about 2 px/s; clear approach to nearby structure is 5 px/s or more. Turns give hundreds, so it is
    only meaningful together with a small yaw.
    """
    import cv2
    height, width = first_rgb.shape[:2]
    size = (WORK_WIDTH, int(round(height * WORK_WIDTH / width))) if width > WORK_WIDTH else (width, height)
    gray = [cv2.cvtColor(cv2.resize(frame, size) if size != (width, height) else frame, cv2.COLOR_RGB2GRAY)
            for frame in (first_rgb, last_rgb)]
    detector = cv2.SIFT_create(nfeatures=2000)
    ka, da = detector.detectAndCompute(gray[0], None)
    kb, db = detector.detectAndCompute(gray[1], None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return None
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 20:
        return None
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    return round(float(np.median(np.linalg.norm(pb - pa, axis=1))) / seconds, 1)


def chunk_motion(frames, intrinsics, previous_last=None):
    """Summarize measured motion over one decoded chunk (uint8 RGB frames, shape [T,H,W,3]).

    previous_last: last frame of the previous chunk, so the estimate covers the whole second.
    """
    sequence = list(frames) if previous_last is None else [previous_last] + list(frames)
    indices = list(range(0, len(sequence), STEP))
    if indices[-1] != len(sequence) - 1 and len(sequence) - 1 - indices[-1] >= 2:
        indices.append(len(sequence) - 1)
    pairs = [pair_motion(sequence[a], sequence[b], intrinsics) for a, b in zip(indices[:-1], indices[1:])]
    usable = [p for p in pairs if p["usable"]]
    seconds = (len(sequence) - 1) / 16
    result = {"pairs": len(pairs), "reliable_pairs": len(usable), "seconds": seconds}
    if len(usable) >= MIN_PAIRS_FOR_YAW:
        yaw = sum(p["yaw_degrees"] for p in usable) * len(pairs) / len(usable)
        result["yaw_degrees"] = 0.0 if abs(yaw) < 1.0 else yaw
    flow = image_motion(sequence[0], sequence[-1], seconds)
    if flow is not None:
        result["image_motion_px_per_second"] = flow
    return result


def feedback_note(measure, commanded_action, commanded_yaw_degrees):
    """Compact player-facing note for one chunk; unreliable quantities are reported as unavailable."""
    note = {"commanded_action": commanded_action,
            "commanded_yaw_degrees_this_chunk": round(commanded_yaw_degrees, 1),
            "reliable_pairs": f"{measure['reliable_pairs']}/{measure['pairs']}"}
    note["measured_yaw_degrees_this_chunk"] = round(measure["yaw_degrees"], 1) if "yaw_degrees" in measure else "unavailable"
    note["image_motion_px_per_second"] = measure.get("image_motion_px_per_second", "unavailable")
    return note


class MotionLog:
    """Accumulates chunk measurements and renders the cumulative summary for the player."""

    def __init__(self, intrinsics):
        self.intrinsics = intrinsics
        self.previous_last = None
        self.chunks = []
        self.cumulative_yaw = 0.0
        self.missed_chunks = 0

    def add(self, frames, commanded_action, commanded_yaw_degrees):
        measure = chunk_motion(frames, self.intrinsics, self.previous_last)
        self.previous_last = frames[-1]
        if "yaw_degrees" in measure:
            self.cumulative_yaw += measure["yaw_degrees"]
        else:
            self.missed_chunks += 1
        entry = {"action": commanded_action, "commanded_yaw_degrees": commanded_yaw_degrees, **measure}
        self.chunks.append(entry)
        return entry

    def summary(self, recent=3):
        if not self.chunks:
            return None
        history = []
        for index, entry in enumerate(self.chunks[-recent:], start=max(0, len(self.chunks) - recent)):
            history.append({"chunk": index, **feedback_note(entry, entry["action"], entry["commanded_yaw_degrees"])})
        return {"scope": "Visual-odometry estimates from the generated frames: noisy, monocular, unavailable when "
                         "evidence is weak. Positive yaw = right. image_motion_px_per_second is the median feature "
                         "displacement between the chunk's first and last frame (about 1 when nothing moves, about 2 for "
                         "a step toward a distant landmark, 5 or more for clear approach to nearby structure; hundreds "
                         "during turns): a low value after a movement command means the world did not visibly move.",
                "recent_chunks": history,
                "cumulative_measured_yaw_degrees": round(self.cumulative_yaw, 1),
                "chunks_without_yaw_measurement": self.missed_chunks}
