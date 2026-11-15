import numpy as np
import pytest

from worldsync.lingbot_controls import CameraPlan
from worldsync.lingbot_paths import PATHS, native_frame_poses


@pytest.mark.parametrize("name", list(PATHS))
def test_fixed_paths_close_and_native_keyframes_preserve_online_camera_poses(name):
    actions = PATHS[name]
    native, final = native_frame_poses(actions)
    camera = CameraPlan()
    for action in actions:
        camera.advance(action)
    assert len(native) == len(actions) * 16 - 3
    np.testing.assert_allclose(native[::4], np.stack(camera.poses), atol=1e-12)
    np.testing.assert_allclose(final["position_normalized_units"], [0, 0, 0], atol=1e-12)
    assert final["heading_degrees"] == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize("name,sign", [("turn90_wait_return", 1), ("turn90left_wait_return", -1)])
def test_task_matched_paths_reach_exactly_ninety_degrees_and_return(name, sign):
    native, final = native_frame_poses(PATHS[name], turn_degrees_per_second=45)
    turned = native[11 * 4]
    yaw = np.degrees(np.arctan2(turned[0, 2], turned[2, 2]))
    assert yaw == pytest.approx(sign * 90, abs=1e-10)
    assert final["heading_degrees"] == pytest.approx(0, abs=1e-10)
    from worldsync.lingbot_diagnostics import summarize
    diagnostics = summarize(PATHS[name], turn_degrees_per_second=45)
    assert diagnostics["total_turn_degrees"] == pytest.approx(180)
