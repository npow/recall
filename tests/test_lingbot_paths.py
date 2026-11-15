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
