import numpy as np
import pytest

pytest.importorskip("torch")
from worldsync.lingbot_budgetmem import evict_redundant, select_by_pose
from worldsync.lingbot_controls import CameraPlan


def track(actions):
    cam = CameraPlan(turn_degrees_per_second=68)
    for a in actions:
        cam.advance(a)
    return np.stack(cam.poses)


def test_selects_turnout_frames_when_returning():
    poses = track(["WAIT"] + ["RIGHT"] * 3 + ["WAIT"] * 6 + ["LEFT"] * 2)
    first_frame = len(poses) - 4
    chosen = select_by_pose(poses, set(range(6, first_frame - 4)), first_frame, 6, 6)
    assert len(chosen) == 6 and all(6 <= i < first_frame - 4 for i in chosen)
    # nearest to a heading about 68 degrees must come from the turn-out, not the long hold
    assert min(chosen) < 16


def test_eviction_bounds_and_keeps_diverse_poses():
    poses = track(["WAIT"] + ["RIGHT"] * 3 + ["WAIT"] * 6 + ["LEFT"] * 3)
    keep = evict_redundant(poses, set(range(6, len(poses) - 4)), 12, len(poses) - 4)
    assert len(keep) == 12
    hold = [i for i in keep if 16 <= i < 40]
    assert len(hold) < 6  # the identical-pose hold is thinned first
