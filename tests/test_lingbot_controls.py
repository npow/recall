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


def test_yaw_clamp_lands_exactly_on_the_limit_and_holds():
    import math
    camera = CameraPlan(50, yaw_limits_degrees=(0, 360))
    for _ in range(12):
        camera.advance("RIGHT")
    assert math.degrees(camera.yaw_commanded) == pytest.approx(360)
    assert camera.commanded_state()["heading_degrees"] == pytest.approx(0, abs=1e-6)
    back = CameraPlan(50, yaw_limits_degrees=(-90, 0))
    for action in ["LEFT"] * 3 + ["RIGHT"] * 4:
        back.advance(action)
    assert math.degrees(back.yaw_commanded) == pytest.approx(0, abs=1e-9)


def test_plan_yaw_limits_from_reference_plan():
    from worldsync.lingbot_controls import plan_yaw_limits
    assert plan_yaw_limits({"prompt": "Rotate the camera 360 degrees", "action_sequence_steps": ["hold(Right,7200ms)"]}) == (0.0, 360.0)
    assert plan_yaw_limits({"prompt": "Turn left 90 degrees then back", "action_sequence_steps": ["hold(Left,1800ms)", "hold(Right,1800ms)"]}) == (-90.0, 0.0)
    assert plan_yaw_limits({"prompt": "Walk then turn around", "action_sequence_steps": ["hold(W,7200ms)", "hold(S,7200ms)"]}) is None


def test_circle_actions_orbit_the_point_ahead_and_keep_facing_it():
    camera = CameraPlan(orbit_radius=4.0, orbit_degrees_per_second=45)
    camera.advance("WAIT")
    centre = np.array([0.0, 0.0, 4.0, 1.0])
    start = camera.poses[-1].copy()
    for _ in range(8):
        camera.advance("CIRCLE_LEFT")
        pose = camera.poses[-1]
        np.testing.assert_allclose(np.linalg.inv(pose) @ centre, centre, atol=1e-9)  # still 4 units straight ahead
        np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(camera.poses[-1], start, atol=1e-9)  # a full circle in 8 chunks
    assert camera.commanded_state()["accumulated_yaw_degrees"] == pytest.approx(360)
    camera = CameraPlan()
    camera.advance("WAIT")
    camera.advance("CIRCLE_LEFT")
    state = camera.commanded_state()
    assert state["heading_degrees"] == pytest.approx(45)
    assert state["position_normalized_units"][0] < 0 and state["position_normalized_units"][2] > 0  # moved left/forward
    mirror = CameraPlan()
    mirror.advance("WAIT")
    mirror.advance("CIRCLE_RIGHT")
    assert mirror.commanded_state()["heading_degrees"] == pytest.approx(-45)
    assert mirror.commanded_state()["position_normalized_units"][0] == pytest.approx(-state["position_normalized_units"][0])
