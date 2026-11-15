import numpy as np

from worldsync.revisit_pairs import find_revisits, overlap


def pose(position, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    p = np.eye(4)
    p[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    p[:3, 3] = position
    return p


def loop_trajectory(n=480):
    """Walk out along +x, turn around, walk back: revisits the start pose heading."""
    poses = []
    for t in range(n):
        if t < n // 2:
            poses.append(pose([t * 0.05, 0, 0], np.pi / 2))      # forward along +x, looking +x
        else:
            poses.append(pose([(n - t) * 0.05, 0, 0], np.pi / 2))  # walk back, same heading
    return np.stack(poses)


def test_straight_line_has_no_revisit():
    poses = np.stack([pose([t * 0.05, 0, 0], np.pi / 2) for t in range(480)])
    assert find_revisits(poses) == []


def test_out_and_back_is_found_and_left_view():
    pairs = find_revisits(loop_trajectory(), min_gap_s=8, max_gap_s=30, radius_frac=0.05)
    assert pairs, "an out-and-back path must contain a revisit"
    p = pairs[0]
    assert p["gap_s"] >= 8 and p["overlap"] >= 0.55 and p["min_overlap_between"] <= 0.12


def test_overlap_decays_with_distance_and_angle():
    poses = np.stack([pose([0, 0, 0], 0.0), pose([0, 0, 0], 0.0), pose([5, 0, 0], 0.0), pose([0, 0, 0], np.pi)])
    ov = overlap(poses, 0, np.array([1, 2, 3]), radius=1.0)
    assert ov[0] > 0.99 and ov[1] < 0.01 and ov[2] < 0.01


def test_static_camera_yields_nothing():
    assert find_revisits(np.stack([np.eye(4)] * 400)) == []
