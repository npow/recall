import cv2
import numpy as np

from scripts.check_lingbot_geometry import fit_pose, unproject


def test_geometry_localization_recovers_known_camera_with_outliers():
    rng = np.random.default_rng(91)
    world = rng.uniform([-2, -1, 4], [2, 1, 8], (400, 3))
    intrinsics = np.array([[330., 0, 252], [0, 330, 140], [0, 0, 1]])
    rotation = np.array([.02, .15, -.01])
    translation = np.array([.15, -.03, .08])
    pixels = cv2.projectPoints(world, rotation, translation, intrinsics, None)[0].reshape(-1, 2)
    pixels += rng.normal(0, .2, pixels.shape)
    pixels[:80] = rng.uniform([0, 0], [504, 280], (80, 2))
    report, pose = fit_pose(world, pixels, intrinsics, (280, 504))
    assert report["usable"]
    assert report["withheld_fraction_within3px"] > .7
    np.testing.assert_allclose(pose[:, :3], cv2.Rodrigues(rotation)[0], atol=.005)
    np.testing.assert_allclose(pose[:, 3], translation, atol=.02)


def test_geometry_gate_rejects_unrelated_images():
    rng = np.random.default_rng(19)
    world = rng.uniform([-2, -1, 4], [2, 1, 8], (200, 3))
    pixels = rng.uniform([0, 0], [504, 280], (200, 2))
    intrinsics = np.array([[330., 0, 252], [0, 330, 140], [0, 0, 1]])
    report, _ = fit_pose(world, pixels, intrinsics, (280, 504))
    assert not report["usable"]


def test_depth_backprojection_uses_world_to_camera_convention():
    # A camera displaced2units right observes this world point at its center.
    intrinsics = np.array([[330., 0, 252], [0, 330, 140], [0, 0, 1]])
    extrinsics = np.column_stack([np.eye(3), [-2., 0, 0]])
    world = unproject(np.array([[252., 140.]]), np.array([5.]), intrinsics, extrinsics)
    np.testing.assert_allclose(world, [[2., 0, 5.]])


def test_shared_feature_holdout_never_influences_fitted_camera():
    rng = np.random.default_rng(127)
    world = rng.uniform([-2, -1, 4], [2, 1, 8], (400, 3))
    intrinsics = np.array([[330., 0, 252], [0, 330, 140], [0, 0, 1]])
    pixels = cv2.projectPoints(world, np.array([.02, .15, -.01]),
                              np.array([.15, -.03, .08]), intrinsics, None)[0].reshape(-1, 2)
    ids = np.arange(400, dtype=np.uint64)
    held_out = ((ids * 2654435761) % (2 ** 32)) < (2 ** 32 // 3)
    good, good_pose = fit_pose(world, pixels, intrinsics, (280, 504), feature_ids=ids)
    corrupted = pixels.copy()
    corrupted[held_out] += [70, 50]
    bad, bad_pose = fit_pose(world, corrupted, intrinsics, (280, 504), feature_ids=ids)
    assert good["usable"] and not bad["usable"]
    assert good["fit_inliers"] == bad["fit_inliers"]
    np.testing.assert_array_equal(good_pose, bad_pose)
