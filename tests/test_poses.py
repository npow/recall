import numpy as np
import pytest

from worldsync.poses import (
    LATENT_SECONDS,
    CameraRig,
    actions_to_latent_steps,
    horizontal_fov,
    parse_actions,
    relative_poses,
    scaled_intrinsics,
)


def rollout(text):
    rig = CameraRig()
    return [rig.advance(keys, seconds) for keys, seconds in actions_to_latent_steps(parse_actions(text))]


def yaw_degrees(pose):
    forward = pose[:3, 2]
    return np.degrees(np.arctan2(forward[0], forward[2]))


def test_parses_playworld_action_strings():
    actions = parse_actions("hold(Left,1800ms); hold(W, 450ms) wait(900ms)")
    assert [(a.keys, a.seconds) for a in actions] == [(("LEFT",), 1.8), (("W",), 0.45), ((), 0.9)]
    with pytest.raises(ValueError):
        parse_actions("hold(Q,100ms)")


def test_left_turn_matches_playworld_timing_and_direction():
    poses = rollout("hold(Left,1800ms)")
    assert len(poses) == round(1.8 / LATENT_SECONDS) == 7
    assert yaw_degrees(poses[-1]) == pytest.approx(-90)
    assert np.allclose(poses[-1][:3, 3], 0)


def test_turn_and_return_closes_exactly():
    poses = rollout("hold(Left,1800ms); hold(Right,1800ms)")
    assert np.allclose(poses[-1], np.eye(4))


def test_full_turn_is_360_degrees():
    poses = rollout("hold(Right,7200ms)")
    assert np.allclose(poses[-1], np.eye(4), atol=1e-9)
    assert len(poses) == 29
    assert yaw_degrees(poses[0]) == pytest.approx(360 / 29)


def test_walking_moves_along_current_heading():
    rig = CameraRig()
    rig.advance(("RIGHT",), seconds=1.8)
    pose = rig.advance(("W",))
    assert np.allclose(pose[:3, 3], [1, 0, 0])


def test_looking_up_never_accumulates_roll():
    rig = CameraRig()
    rig.advance(("UP",), seconds=0.4)
    pose = rig.advance(("LEFT",), seconds=1.8)
    right = pose[:3, 0]
    assert right[1] == pytest.approx(0, abs=1e-12)


def test_relative_poses_follow_lingbot_framewise_convention():
    poses = np.stack(rollout("hold(W,1000ms); hold(Left,500ms)"))
    first = relative_poses(poses[:4], None)
    assert np.allclose(first[0], np.eye(4))
    assert np.allclose(first[1][:3, 3], [0, 0, 1])
    later = relative_poses(poses[4:], poses[3])
    assert np.allclose(later[0][:3, :3], (np.linalg.inv(poses[3]) @ poses[4])[:3, :3])


def test_intrinsics_scale_with_resolution():
    assert horizontal_fov() == pytest.approx(90.0, abs=0.1)
    assert np.allclose(scaled_intrinsics(960, 1664), [831.2, 831.2, 832.0, 480.0])
