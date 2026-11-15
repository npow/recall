"""Training-free keyframe-restart memory for LingBot-World2 (diagnostic, default off).

Idea: keep real generated keyframes together with the commanded camera pose at the instant they were
shown. When the commanded pose re-enters a stored keyframe's neighbourhood after a long absence (longer
than the rolling KV window), restart image-to-video generation from the stored keyframe: new
`pipeline.generate` call with the keyframe as conditioning image, fresh KV caches, and the native pose
track continuing from the current pose. The revisit therefore matches the stored view by construction,
which is the point and also the cheating risk: dynamic state that changed since the keyframe is lost.

Bookkeeping (native-fixed-control paths). Native frame n of the pose track is latent frame n/4. A chunk
ends on native frame 16c+12 (chunk 0 has 13 decoded frames, later chunks 16). A restart at the start of
global chunk g conditions on the keyframe image at the NEXT latent step (native 16g, scheme B, default), so the
restart segment stays aligned with the commanded timeline; the remaining segment produces (maximum - g)
chunks, keeping the total chunk count at `maximum`. Scheme A (conditioning at native 16g-4) is kept for the
first runs; it lags the commanded timeline by one latent frame per restart.
"""
from dataclasses import dataclass, field

import numpy as np

NATIVE_FRAMES_PER_CHUNK = 16


@dataclass
class Keyframe:
    native_index: int          # native pose-track index at which the keyframe was shown
    pose: np.ndarray           # 4x4 camera-to-world pose at that instant
    pixels: np.ndarray         # HxWx3 uint8 decoded frame
    label: str = ""


def pose_distance(pose_a, pose_b):
    """Return (translation distance in normalized units, rotation angle in degrees)."""
    translation = float(np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3]))
    relative = pose_a[:3, :3].T @ pose_b[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return translation, float(np.degrees(np.arccos(cosine)))


def excursion(native_poses, start, end, reference_pose):
    """Largest translation (units) and rotation (degrees) separation from `reference_pose` over native frames [start, end]."""
    track = np.asarray(native_poses)[int(start):int(end) + 1:4]
    if len(track) == 0:
        return 0.0, 0.0
    translation = float(np.max(np.linalg.norm(track[:, :3, 3] - reference_pose[:3, 3], axis=1)))
    relative = np.einsum("ji,njk->nik", reference_pose[:3, :3], track[:, :3, :3])
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return translation, float(np.degrees(np.arccos(cosine)).max())


def gray_small(pixels, size=(112, 64)):
    """Cheap grayscale thumbnail used for the drift check (no DINO dependency in the rollout)."""
    from PIL import Image
    image = Image.fromarray(np.asarray(pixels)).convert("L").resize(size)
    return np.asarray(image, dtype=np.float32)


def ncc(pixels_a, pixels_b):
    a, b = gray_small(pixels_a), gray_small(pixels_b)
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float((a * b).sum() / denominator) if denominator > 0 else 0.0


@dataclass
class KeyframeBank:
    position_threshold: float = 1.0
    heading_threshold_degrees: float = 15.0
    min_gap_native: int = 5 * NATIVE_FRAMES_PER_CHUNK    # beyond the 18-latent-frame window (4.5 chunks)
    cooldown_native: int = 5 * NATIVE_FRAMES_PER_CHUNK
    drift_threshold: float = 2.0                          # restart only if NCC(current, keyframe) < this
    min_excursion_degrees: float = 0.0                    # a revisit needs the camera to have LEFT the keyframe pose by this much ...
    min_excursion_position: float = 0.0                   # ... or by this many units since the keyframe was shown (0/0 disables)
    frames: list = field(default_factory=list)
    restarts: list = field(default_factory=list)
    last_restart_native: int = -10 ** 9

    def add(self, native_index, pose, pixels, label=""):
        self.frames.append(Keyframe(int(native_index), np.array(pose, dtype=np.float64), pixels, label))

    def find(self, native_now, pose_now, current_pixels=None, native_poses=None):
        """Best keyframe whose pose is close to `pose_now` and which is older than the cache window."""
        if native_now - self.last_restart_native < self.cooldown_native:
            return None
        best, best_score = None, None
        for keyframe in self.frames:
            if native_now - keyframe.native_index < self.min_gap_native:
                continue
            translation, angle = pose_distance(pose_now, keyframe.pose)
            if translation > self.position_threshold or angle > self.heading_threshold_degrees:
                continue
            if native_poses is not None and (self.min_excursion_degrees > 0 or self.min_excursion_position > 0):
                moved, turned = excursion(native_poses, keyframe.native_index, native_now, keyframe.pose)
                if moved < self.min_excursion_position and turned < self.min_excursion_degrees:
                    continue            # still on the same continuous view (e.g. a hold), not a revisit
            if current_pixels is not None and ncc(current_pixels, keyframe.pixels) >= self.drift_threshold:
                continue
            score = translation / max(self.position_threshold, 1e-9) + angle / max(self.heading_threshold_degrees, 1e-9)
            if best_score is None or score < best_score:
                best, best_score = keyframe, score
        return best

    def record_restart(self, chunk, native_now, keyframe, current_pixels):
        translation_gap = None
        self.last_restart_native = int(native_now)
        entry = {"global_chunk": int(chunk), "native_index": int(native_now), "keyframe_native_index": keyframe.native_index,
                 "keyframe_label": keyframe.label, "gap_native_frames": int(native_now - keyframe.native_index),
                 "ncc_before_restart": None if current_pixels is None else ncc(current_pixels, keyframe.pixels)}
        self.restarts.append(entry)
        return entry


def segment_poses(native_poses, native_now, aligned=True):
    """Pose track for the restart segment.

    aligned=True (scheme B): the conditioning frame sits one latent step ahead (native_now + 4), so the
    segment stays on the commanded timeline; the skipped step shows up as 3 missing native frames in the video.
    aligned=False (scheme A, used in the first runs): condition at native_now, which leaves the camera one latent
    frame (up to 17 degrees of a turn) behind the commanded pose after every restart.
    """
    return np.asarray(native_poses)[int(native_now) + (4 if aligned else 0):]


def native_last_index(generated_frames, restarts, aligned=True):
    """Native pose-track index of the last emitted video frame."""
    return generated_frames - 1 + (3 * restarts if aligned else -restarts)


def crossfade(previous_frame, new_frames, count):
    """Blend the first `count` frames of a restart chunk from the previous view into the keyframe view."""
    if count <= 0:
        return new_frames
    blended = new_frames.copy()
    previous = previous_frame.astype(np.float32)
    for i in range(min(count, len(new_frames))):
        alpha = (i + 1) / (count + 1)
        blended[i] = np.clip((1 - alpha) * previous + alpha * new_frames[i].astype(np.float32), 0, 255).astype(np.uint8)
    return blended
