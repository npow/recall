"""Fixed OpenCV camera-control mapping for the reconstructed LingBot adapter."""
import math
import numpy as np


class CameraPlan:
    def __init__(self, turn_degrees_per_second=60):
        self.poses = []
        self.turn = math.radians(turn_degrees_per_second) / 4
        self.yaw_commanded = 0.0

    def advance(self, action):
        step = np.eye(4)
        translations = {"W": [0, 0, 1], "S": [0, 0, -1], "A": [-1, 0, 0], "D": [1, 0, 0]}
        if action in translations:
            step[:3, 3] = translations[action]
        elif action in {"LEFT", "RIGHT"}:
            angle = self.turn if action == "RIGHT" else -self.turn
            c, s = math.cos(angle), math.sin(angle)
            step[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
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
        if action in {"LEFT", "RIGHT"}:
            self.yaw_commanded += count * (self.turn if action == "RIGHT" else -self.turn)
        return np.stack(self.poses)

    def commanded_state(self):
        pose = self.poses[-1] if self.poses else np.eye(4)
        return {"position_normalized_units": pose[:3, 3].tolist(),
                "heading_degrees": math.degrees(math.atan2(pose[0, 2], pose[2, 2])),
                "accumulated_yaw_degrees": math.degrees(self.yaw_commanded),
                "world_time_seconds": max(0, len(self.poses) - 1) / 4,
                "scope": "Intended camera commands only; not measured visual motion or goal completion."}
