# Rejected simple visual-rotation feedback

Before using measured camera state to correct yaw drift, fit SIFT feature correspondences to a rotating camera under the model's nominal intrinsics. The implementation rejects weak correspondence, low image coverage, or a homography inconsistent with rotation by more than3 pixels over the field of view. Synthetic projection tests recover known positive/negative12-degree turns and reject empty images and anisotropic warps.

On the GC007 seed42 delayed-return memory video, all24 moving image pairs are rejected. Median inlier count is119.5, median inlier fraction0.681, median image coverage0.318, but median rotation-fit residual is8.61pixels. The39 accepted pairs are stationary holds. We therefore do not use this estimator for automatic control corrections or as ground-truth pose recovery.

This does not prove which part of the generated camera behavior is wrong: nominal intrinsics may not describe the initial photograph, and translation, parallax or nonrigid generation can invalidate the pure-rotation assumption. The result rules out treating this simple nominal-calibration fit as a reliable visual gyro. Camera overshoot remains an explicit limitation of the memory method. The full per-pair record is preserved in motion.json; diagnostic code is in scripts/measure_camera_motion.py and worldsync/lingbot_visual_pose.py.
