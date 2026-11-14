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


def test_pose_trace_distinguishes_backward_translation_turns_and_wait():
    camera = CameraPlan()
    camera.advance("WAIT")
    camera.advance("W")
    forward = camera.commanded_state()
    assert forward["position_normalized_units"] == [0, 0, 4]
    camera.advance("S")
    assert camera.commanded_state()["heading_degrees"] == 0
    np.testing.assert_allclose(camera.commanded_state()["position_normalized_units"], [0, 0, 0])
    for _ in range(3):
        camera.advance("RIGHT")
    turned = camera.commanded_state()
    assert turned["accumulated_yaw_degrees"] == pytest.approx(180)
    assert abs(turned["heading_degrees"]) == pytest.approx(180)
    camera.advance("WAIT")
    held = camera.commanded_state()
    assert held["position_normalized_units"] == turned["position_normalized_units"]
    assert held["heading_degrees"] == turned["heading_degrees"]
    assert held["world_time_seconds"] == turned["world_time_seconds"] + 1
