import cv2
import numpy as np
import pytest

from worldsync.lingbot_visual_pose import estimate_rotation


@pytest.mark.parametrize("degrees", [-12, 12])
def test_recovers_known_camera_turn_from_projected_textured_scene(degrees):
    rng = np.random.default_rng(11)
    image = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), .6)
    intrinsics = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]])
    angle = np.deg2rad(degrees)
    camera = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    projected = cv2.warpPerspective(image, intrinsics @ camera.T @ np.linalg.inv(intrinsics), (640, 480))
    result = estimate_rotation(image, projected, intrinsics)
    assert result["usable"]
    assert result["apparent_yaw_degrees"] == pytest.approx(degrees, abs=.5)


def test_refuses_to_invent_pose_without_visual_correspondences():
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    assert not estimate_rotation(image, image, np.eye(3))["usable"]


def test_rejects_visible_anisotropic_warp_as_camera_rotation():
    rng = np.random.default_rng(12)
    image = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    intrinsics = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]])
    warped = cv2.warpPerspective(image, np.array([[1.2, 0, -64], [0, .8, 48], [0, 0, 1]]), (640, 480))
    result = estimate_rotation(image, warped, intrinsics)
    assert not result["usable"]
    assert result["rotation_fit_residual_px"] > 3
