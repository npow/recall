import numpy as np
import pytest

from worldsync.lingbot_controls import CameraPlan


def test_neutral_controls_remain_identity_across_chunk_boundaries():
    camera = CameraPlan()
    assert camera.advance("WAIT").shape == (4, 4, 4)
    poses = camera.advance("WAIT")
    assert poses.shape == (8, 4, 4)
    np.testing.assert_array_equal(poses, np.repeat(np.eye(4)[None], 8, axis=0))


def test_opencv_turns_and_translation_use_the_local_camera_axes():
    camera = CameraPlan()
    poses = camera.advance("RIGHT")
    forward = poses[-1, :3, 2]
    assert forward[0] > 0 and forward[2] > 0
    prior = poses[-1].copy()
    poses = camera.advance("W")
    np.testing.assert_allclose(poses[-1, :3, 3] - prior[:3, 3], 4 * forward)
    np.testing.assert_allclose(poses[-1, :3, :3].T @ poses[-1, :3, :3], np.eye(3), atol=1e-12)
    with pytest.raises(ValueError):
        camera.advance("SHELL")
