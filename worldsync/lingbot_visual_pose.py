"""Conservative visual rotation diagnostic, not ground-truth camera recovery.

Fits feature motion to a single rotating camera using the model's nominal K.
Rejects pairs with weak coverage or motion inconsistent with that assumption.
"""
import math
import cv2
import numpy as np


def estimate_rotation(previous_rgb, current_rgb, intrinsics):
    result = {"usable": False, "scope": "apparent rotation under nominal intrinsics and a pure-rotation assumption"}
    detector = cv2.SIFT_create(nfeatures=3000)
    first, desc_first = detector.detectAndCompute(cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2GRAY), None)
    second, desc_second = detector.detectAndCompute(cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY), None)
    if desc_first is None or desc_second is None or len(desc_second) < 2:
        return {**result, "reason": "insufficient features", "matches": 0}
    pairs = cv2.BFMatcher().knnMatch(desc_first, desc_second, k=2)
    matches = [a for a, b in pairs if a.distance < .7 * b.distance]
    result["matches"] = len(matches)
    if len(matches) < 12:
        return {**result, "reason": "insufficient matches"}
    source = np.float32([first[match.queryIdx].pt for match in matches])
    target = np.float32([second[match.trainIdx].pt for match in matches])
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 2.5)
    if homography is None:
        return {**result, "reason": "homography failed"}
    mask = mask.ravel().astype(bool)
    height, width = previous_rgb.shape[:2]
    count = int(mask.sum())
    fraction = count / len(matches)
    coverage = cv2.contourArea(cv2.convexHull(source[mask])) / (height * width) if count >= 3 else 0
    normalized = np.linalg.inv(intrinsics) @ homography @ intrinsics
    if np.linalg.det(normalized) <= 0:
        return {**result, "reason": "orientation-reversing fit"}
    normalized /= np.cbrt(np.linalg.det(normalized))
    u, _, vt = np.linalg.svd(normalized)
    correction = np.diag([1, 1, np.linalg.det(u @ vt)])
    image_rotation = u @ correction @ vt
    rotation_homography = intrinsics @ image_rotation @ np.linalg.inv(intrinsics)
    grid = np.float32([[[x, y] for x in (.1 * width, .5 * width, .9 * width)
                        for y in (.1 * height, .5 * height, .9 * height)]])
    fitted = cv2.perspectiveTransform(grid, homography)
    rotational = cv2.perspectiveTransform(grid, rotation_homography)
    residual = float(np.median(np.linalg.norm(fitted - rotational, axis=-1)))
    camera_rotation = image_rotation.T
    yaw = math.degrees(math.atan2(camera_rotation[0, 2], camera_rotation[2, 2]))
    usable = count >= 30 and fraction >= .5 and coverage >= .1 and residual <= 3
    return {**result, "usable": usable, "reason": "accepted" if usable else "weak or non-rotational visual evidence",
            "inliers": count, "inlier_fraction": fraction, "inlier_coverage": coverage,
            "rotation_fit_residual_px": residual, "apparent_yaw_degrees": yaw,
            "relative_camera_rotation": camera_rotation.tolist()}
