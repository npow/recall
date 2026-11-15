"""Predeclared camera calibration paths, independent of player decisions."""
import numpy as np

from worldsync.lingbot_controls import CameraPlan

PATHS = {
    "turn90_wait_return": ["WAIT"] + ["RIGHT"] * 2 + ["WAIT"] * 8 + ["LEFT"] * 2 + ["WAIT"],
    "turn90left_wait_return": ["WAIT"] + ["LEFT"] * 2 + ["WAIT"] * 8 + ["RIGHT"] * 2 + ["WAIT"],
    "turn_return": ["WAIT"] + ["RIGHT"] * 3 + ["LEFT"] * 3 + ["WAIT"],
    "turn_hold_return": ["WAIT"] + ["RIGHT"] * 3 + ["WAIT"] * 6 + ["LEFT"] * 3 + ["WAIT"],  # revisit matrix / keyframe-restart path
    "turn_long_wait_return": ["WAIT"] + ["RIGHT"] * 3 + ["WAIT"] * 24 + ["LEFT"] * 3 + ["WAIT"],
    "turn_wait_return": ["WAIT"] + ["RIGHT"] * 3 + ["WAIT"] * 8 + ["LEFT"] * 3 + ["WAIT"],
    "translate_return": ["WAIT"] + ["W"] * 4 + ["S"] * 4 + ["WAIT"],
    "spin360": ["WAIT"] + ["RIGHT"] * 6 + ["WAIT"] * 2,
    "trio_viewer": ["WAIT"] + ["RIGHT"] * 3 + ["LEFT"] * 3 + ["WAIT"],
    # Two agents circling each other: A orbits the point 4 units ahead (one full circle, 45 deg/s); B first walks to
    # the opposite side of that circle and turns to face it, then orbits the same way, through A's earlier poses.
    "circle_a": ["WAIT"] + ["CIRCLE_LEFT"] * 8 + ["WAIT"],
    "circle_b": ["WAIT", "W", "W", "RIGHT", "RIGHT", "RIGHT"] + ["CIRCLE_LEFT"] * 4,
    "trio_duel_a": ["WAIT"] + ["S"] * 6,
    "trio_duel_b": ["WAIT"] + ["RIGHT"] * 3 + ["S"] * 4,
    "spin720": ["WAIT"] + ["RIGHT"] * 12 + ["WAIT"],
    "round_trip": ["WAIT"] + ["W"] * 4 + ["RIGHT"] * 3 + ["W"] * 4 + ["RIGHT"] * 3 + ["WAIT"],
}


def native_frame_poses(actions, turn_degrees_per_second=60):
    """Expand quarter-second latent poses into the native 16-FPS input format.

    Native inference samples poses at RGB indices 0,4,8,..., so these preserve
    the specified latent poses exactly up to interpolation rounding.
    """
    from scipy.spatial.transform import Rotation, Slerp
    camera = CameraPlan(turn_degrees_per_second=turn_degrees_per_second)
    for action in actions:
        camera.advance(action)
    latent = np.stack(camera.poses)
    knots = np.arange(len(latent)) * 4
    times = np.arange(knots[-1] + 1)
    poses = np.repeat(np.eye(4)[None], len(times), axis=0)
    poses[:, :3, :3] = Slerp(knots, Rotation.from_matrix(latent[:, :3, :3]))(times).as_matrix()
    for axis in range(3):
        poses[:, axis, 3] = np.interp(times, knots, latent[:, axis, 3])
    return poses, camera.commanded_state()
