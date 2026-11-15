"""Fixed OpenCV camera-control mapping for the reconstructed LingBot adapter."""
import math
import re
import numpy as np


def fp32_camera_rays(builder, poses, intrinsics, height, width):
    """Match native ray computation even when called inside denoiser autocast."""
    import torch
    with torch.amp.autocast("cuda", enabled=False):
        return builder(poses, intrinsics, height, width)


class CameraPlan:
    def __init__(self, turn_degrees_per_second=60, move_speed=1.0, yaw_limits_degrees=None, orbit_radius=4.0,
                 orbit_degrees_per_second=45):
        # The native helper rescales translations to unit maximum per call, so
        # move_speed only matters relative to rotation (and is 1 in stock).
        self.poses = []
        self.speed = move_speed
        self.turn = math.radians(turn_degrees_per_second) / 4
        # CIRCLE_LEFT/CIRCLE_RIGHT orbit the point ``orbit_radius`` ahead while staying turned toward it
        # (two agents circling each other): CIRCLE_LEFT moves the camera to its left and turns it right.
        self.orbit_radius = orbit_radius
        self.orbit = math.radians(orbit_degrees_per_second) / 4
        self.yaw_commanded = 0.0
        # Optional (lo, hi) range for the commanded yaw in degrees (right positive). A turn that would pass a limit is
        # shortened so the yaw lands exactly on it; the camera path has four pose steps per chunk, so this is finer
        # than the one-second action granularity.
        self.yaw_limits = None if yaw_limits_degrees is None else tuple(math.radians(v) for v in yaw_limits_degrees)

    def advance(self, action):
        step = np.eye(4)
        translations = {"W": [0, 0, 1], "S": [0, 0, -1], "A": [-1, 0, 0], "D": [1, 0, 0]}
        if action in translations:
            step[:3, 3] = np.array(translations[action]) * self.speed
        elif action in {"LEFT", "RIGHT"}:
            angle = self.turn if action == "RIGHT" else -self.turn
            if self.yaw_limits is not None:
                steps = 3 if not self.poses else 4
                target = min(max(self.yaw_commanded + steps * angle, self.yaw_limits[0]), self.yaw_limits[1])
                angle = (target - self.yaw_commanded) / steps
            c, s = math.cos(angle), math.sin(angle)
            step[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        elif action in {"CIRCLE_LEFT", "CIRCLE_RIGHT"}:
            angle = self.orbit if action == "CIRCLE_LEFT" else -self.orbit
            c, s = math.cos(angle), math.sin(angle)
            step[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
            step[:3, 3] = [-self.orbit_radius * s, 0, self.orbit_radius * (1 - c)]  # centre stays at (0, 0, r)
        elif action in {"UP", "DOWN"}:
            angle = self.turn if action == "UP" else -self.turn
            c, s = math.cos(angle), math.sin(angle)
            step[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
        elif action != "WAIT":
            raise ValueError(f"unsupported camera action: {action}")
        if not self.poses:
            self.poses.append(np.eye(4))
            count = 3
        else:
            count = 4
        for _ in range(count):
            self.poses.append(self.poses[-1] @ step)
        if action in {"LEFT", "RIGHT", "CIRCLE_LEFT", "CIRCLE_RIGHT"}:
            self.yaw_commanded += count * math.atan2(step[0, 2], step[0, 0])
        return np.stack(self.poses)

    def commanded_state(self):
        pose = self.poses[-1] if self.poses else np.eye(4)
        return {"position_normalized_units": pose[:3, 3].tolist(),
                "heading_degrees": math.degrees(math.atan2(pose[0, 2], pose[2, 2])),
                "accumulated_yaw_degrees": math.degrees(self.yaw_commanded),
                "yaw_limits_degrees": None if self.yaw_limits is None else [round(math.degrees(v), 1) for v in self.yaw_limits],
                "world_time_seconds": max(0, len(self.poses) - 1) / 4,
                "scope": "Intended camera commands only; not measured visual motion or goal completion."}


GOAL_ANGLE = re.compile(r"(\d{2,3})\s*-?\s*degree", re.I)


def plan_yaw_limits(task, rate=50.0):
    """Yaw range (lo, hi) in degrees implied by the reference plan's Left/Right holds, or None without turn steps.

    The human plans pace turns at about 50 deg/s. A pure one-direction rotation whose goal states an angle
    ("360 degrees") is limited to that angle; otherwise the range is the envelope of the plan's yaw path.
    """
    from worldsync.reference_player import plan_steps
    total, path, signs = 0.0, [0.0], set()
    for step in plan_steps(task):
        if step["action"] in ("LEFT", "RIGHT"):
            sign = 1 if step["action"] == "RIGHT" else -1
            signs.add(sign)
            total += sign * rate * step["seconds"]
            path.append(total)
    if len(path) == 1:
        return None
    lo, hi = min(path), max(path)
    match = GOAL_ANGLE.search(task.get("prompt", ""))
    if len(signs) == 1 and match:
        angle = float(match.group(1))
        lo, hi = (0.0, angle) if 1 in signs else (-angle, 0.0)
    return (round(lo, 1), round(hi, 1))
