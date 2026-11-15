"""PlayWorld actions to LingBot-World2 camera poses.

Poses are OpenCV camera-to-world matrices (x right, y down, z forward) sampled at
latent-frame rate. LingBot consumes frame-to-frame relative poses whose
translations are divided by a fixed scale, so a full-speed step is 1.0.
"""
from dataclasses import dataclass
import re

import numpy as np

VIDEO_FPS = 16
FRAMES_PER_LATENT = 4
LATENT_SECONDS = FRAMES_PER_LATENT / VIDEO_FPS
# PlayWorld: hold(Left,1800ms) is a 90 degree turn, hold(Right,7200ms) is 360.
ROTATION_DEG_PER_SECOND = 50.0
# World units per latent frame at full walking speed; LingBot sees this as 1.0.
STEP_PER_LATENT = 1.0

TRANSLATION_KEYS = {"W": (0, 0, 1), "S": (0, 0, -1), "A": (-1, 0, 0), "D": (1, 0, 0)}
ROTATION_KEYS = {"LEFT": (-1, 0), "RIGHT": (1, 0), "UP": (0, 1), "DOWN": (0, -1)}
KEY_ALIASES = {"J": "LEFT", "L": "RIGHT", "I": "UP", "K": "DOWN"}
ACTION_PATTERN = re.compile(r"(hold|wait)\(\s*(?:([A-Za-z]+)\s*,\s*)?(\d+(?:\.\d+)?)\s*ms\s*\)", re.I)

# LingBot's example intrinsics at 832x480: fx, fy, cx, cy (about 90 x 60 degree FOV).
DEFAULT_INTRINSICS = np.array([415.6, 415.6, 416.0, 240.0])
INTRINSICS_SIZE = (480, 832)


@dataclass(frozen=True)
class Action:
    keys: tuple[str, ...]
    seconds: float


def parse_actions(text: str) -> list[Action]:
    """Parse PlayWorld strings such as ``hold(Left,1800ms); wait(450ms)``."""
    actions = []
    for kind, key, ms in ACTION_PATTERN.findall(text):
        keys = () if kind.lower() == "wait" else (normalize_key(key),)
        actions.append(Action(keys, float(ms) / 1000))
    if not actions and text.strip():
        raise ValueError(f"no actions found in {text!r}")
    return actions


def normalize_key(key: str) -> str:
    key = key.upper()
    key = KEY_ALIASES.get(key, key)
    if key not in TRANSLATION_KEYS and key not in ROTATION_KEYS:
        raise ValueError(f"unsupported key {key!r}")
    return key


def rotation_y(degrees: float) -> np.ndarray:
    t = np.radians(degrees)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_x(degrees: float) -> np.ndarray:
    t = np.radians(degrees)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


class CameraRig:
    """Integrates held keys into a camera pose, one latent frame at a time.

    Yaw rotates about the world vertical axis, so pitching never accumulates roll.
    """

    def __init__(self, pose: np.ndarray | None = None):
        self.pose = np.eye(4) if pose is None else np.array(pose, dtype=float)

    def advance(self, keys: tuple[str, ...] | list[str], seconds: float = LATENT_SECONDS) -> np.ndarray:
        rotation = self.pose[:3, :3]
        yaw = sum(ROTATION_KEYS[k][0] for k in keys if k in ROTATION_KEYS)
        pitch = sum(ROTATION_KEYS[k][1] for k in keys if k in ROTATION_KEYS)
        degrees = ROTATION_DEG_PER_SECOND * seconds
        # In OpenCV coordinates, turning left moves +z toward -x (negative angle about y),
        # and looking up moves +z toward -y (positive angle about x).
        rotation = rotation_y(yaw * degrees) @ rotation @ rotation_x(pitch * degrees)
        local = np.sum([TRANSLATION_KEYS[k] for k in keys if k in TRANSLATION_KEYS] or [(0, 0, 0)], axis=0)
        norm = np.linalg.norm(local)
        step = STEP_PER_LATENT * seconds / LATENT_SECONDS
        translation = rotation @ (local / norm * step) if norm else np.zeros(3)
        self.pose = self.pose.copy()
        self.pose[:3, :3] = rotation
        self.pose[:3, 3] += translation
        return self.pose.copy()


def actions_to_latent_steps(actions: list[Action]) -> list[tuple[tuple[str, ...], float]]:
    """Expand timed actions into (held keys, seconds) per latent frame.

    Frame boundaries follow the cumulative clock, and each action's exact duration
    is spread over its frames, so a 1800 ms turn is exactly 90 degrees even though
    it spans 7.2 latent frames.
    """
    steps, elapsed, emitted = [], 0.0, 0
    for action in actions:
        elapsed += action.seconds
        frames = max(int(round(elapsed / LATENT_SECONDS)) - emitted, 0)
        steps.extend([(action.keys, action.seconds / frames)] * frames)
        emitted += frames
    return steps


def relative_poses(c2ws: np.ndarray, previous: np.ndarray | None, trans_scale: float = STEP_PER_LATENT) -> np.ndarray:
    """LingBot's framewise relative poses for a run of latent frames.

    ``previous`` is the pose of the latent frame before ``c2ws[0]``; ``None`` marks
    the start of the video, whose relative pose is the identity. Translations are
    divided by ``trans_scale`` (offline LingBot uses the trajectory's largest step).
    """
    c2ws = np.asarray(c2ws, dtype=float)
    prior = np.concatenate([[c2ws[0] if previous is None else previous], c2ws[:-1]])
    relative = np.linalg.inv(prior) @ c2ws
    if previous is None:
        relative[0] = np.eye(4)
    if trans_scale > 0:
        relative[:, :3, 3] /= trans_scale
    return relative


def scaled_intrinsics(height: int, width: int, intrinsics: np.ndarray = DEFAULT_INTRINSICS) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    sy, sx = height / INTRINSICS_SIZE[0], width / INTRINSICS_SIZE[1]
    return np.array([fx * sx, fy * sy, cx * sx, cy * sy])


def horizontal_fov(intrinsics: np.ndarray = DEFAULT_INTRINSICS, width: int = INTRINSICS_SIZE[1]) -> float:
    return float(np.degrees(2 * np.arctan(width / (2 * intrinsics[0]))))
