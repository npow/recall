"""Check that commanded turns and revisits actually happened in the camera path."""
import math
import numpy as np

from worldsync.lingbot_controls import CameraPlan


def summarize(actions, revisit_radius=1.5, min_excursion=4.0):
    """actions: one commanded action per native chunk. Positions are in chunk-step units."""
    camera = CameraPlan()
    chunk_end = []
    for action in actions:
        poses = camera.advance(action)
        chunk_end.append(poses[-1])
    if not chunk_end:
        return {"chunks": 0}
    ends = np.stack(chunk_end)
    position = ends[:, :3, 3][:, [0, 2]]
    forward = ends[:, :3, 2]
    yaw = np.unwrap(np.arctan2(forward[:, 0], forward[:, 2]))
    steps = np.linalg.norm(np.diff(np.vstack([[0, 0], position]), axis=0), axis=1)
    revisit = None
    for i in range(len(position)):
        far = np.linalg.norm(position[:i + 1], axis=1).max()
        if far >= min_excursion and np.linalg.norm(position[i]) <= revisit_radius:
            revisit = i
            break
    held = [i for i, a in enumerate(actions) if a == "WAIT"]
    return {"chunks": len(actions), "path_length": float(steps.sum()),
        "net_yaw_degrees": float(math.degrees(yaw[-1])),
        "total_turn_degrees": float(math.degrees(np.abs(np.diff(np.concatenate([[0.0], yaw]))).sum())),
        "max_distance_from_start": float(np.linalg.norm(position, axis=1).max()),
        "final_distance_from_start": float(np.linalg.norm(position[-1])),
        "revisited_start_at_chunk": revisit, "wait_chunks": len(held),
        "wait_pose_changes": int(sum(np.abs(np.diff(np.vstack([[0, 0], position]), axis=0)[i]).sum() > 1e-9 for i in held))}
