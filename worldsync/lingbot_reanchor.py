"""Start-view re-anchoring diagnostic (training-free).

Archive the canonical KV of the first four latent frames (the initial image and
chunk 0). While the commanded pose is back near the origin pose, place those
frames at recent virtual times inside the same 18-frame attended budget
(sink6 + 4 re-anchor + tail8), so the start view is "recent" to the model
instead of only sitting at absolute time 0 in the sink. Same machinery and
position handling as PoseKVMemory; not a paper reproduction.
"""
import math
import numpy as np

from worldsync.lingbot_retrieval import PoseKVMemory

POSITION_UNITS = 4.0           # one W step is 4 normalized units
HEADING_RADIANS = math.pi / 3  # within 60 degrees of the start heading


def near_origin(pose, position=POSITION_UNITS, heading=HEADING_RADIANS):
    distance = float(np.linalg.norm(pose[:3, 3]))
    angle = math.acos(float(np.clip((np.trace(pose[:3, :3]) - 1) / 2, -1, 1)))
    return distance <= position and angle <= heading


class ReanchorMemory(PoseKVMemory):
    def __init__(self, pipeline):
        super().__init__(pipeline)
        self.archive_range = (0, 4)

    def begin(self, chunk, poses):
        self.chunk = chunk
        self.poses = np.asarray(poses)
        self.cached.clear()
        triggered = near_origin(self.poses[-1]) and set(range(4)) <= self.available
        self.selected = [0, 1, 2, 3] if triggered else []
        self.trace.append({"chunk": chunk, "reanchor": triggered, "retrieved_frame_ids": self.selected,
                           "end_pose_position": self.poses[-1][:3, 3].tolist(), "attention_replaced": False})

    def metadata(self):
        data = super().metadata()
        data["method"] = "start-view re-anchoring (frames 0-3 re-timed as recent); diagnostic"
        data["archive_limit_latent_frames"] = 4
        data["trigger"] = {"position_units": POSITION_UNITS, "heading_degrees": 60}
        return data
