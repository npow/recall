"""Predeclared camera calibration paths, independent of player decisions."""
import numpy as np

from worldsync.lingbot_controls import CameraPlan

PATHS = {
    "turn_return": ["WAIT"] + ["RIGHT"] * 3 + ["LEFT"] * 3 + ["WAIT"],
    "turn_long_wait_return": ["WAIT"] + ["RIGHT"] * 3 + ["WAIT"] * 24 + ["LEFT"] * 3 + ["WAIT"],
    "turn_wait_return": ["WAIT"] + ["RIGHT"] * 3 + ["WAIT"] * 8 + ["LEFT"] * 3 + ["WAIT"],
    "translate_return": ["WAIT"] + ["W"] * 4 + ["S"] * 4 + ["WAIT"],
    "round_trip": ["WAIT"] + ["W"] * 4 + ["RIGHT"] * 3 + ["W"] * 4 + ["RIGHT"] * 3 + ["WAIT"],
}


def native_frame_poses(actions):
    """Expand quarter-second latent poses into the native 16-FPS input format.

    Native inference samples poses at RGB indices 0,4,8,..., so these preserve
    the specified latent poses exactly up to interpolation rounding.
    """
    from scipy.spatial.transform import Rotation, Slerp
    camera = CameraPlan()
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
