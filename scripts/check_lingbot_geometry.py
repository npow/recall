"""Localize unseen later observations against a frozen past-only depth map.

Target RGB is used for camera localization, never to reconstruct or update the
map. Held-out feature matches test whether a fitted pose generalizes. These are
diagnostic estimates, not independently measured ground-truth camera poses.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def resize_like_da3(image, resolution):
    """Match the pinned upper-bound resize then nearest14-pixel resize."""
    height, width = image.shape[:2]
    scale = resolution / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    new = tuple(max(1, int(np.floor(value / 14 + .5)) * 14) for value in size)
    if new != size:
        interpolation = cv2.INTER_CUBIC if any(a > b for a, b in zip(new, size)) else cv2.INTER_AREA
        resized = cv2.resize(resized, new, interpolation=interpolation)
    return resized


def unproject(pixels, depth, intrinsics, extrinsics):
    camera = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(intrinsics).T
    camera *= depth[:, None]
    return (camera - extrinsics[:, 3]) @ extrinsics[:, :3]


def fit_pose(world, pixels, intrinsics, size, feature_ids=None):
    result = {"usable": False, "correspondences": len(world)}
    if len(world) < 45:
        return {**result, "reason": "fewer than45 map correspondences"}, None
    if feature_ids is None:
        order = np.random.default_rng(147).permutation(len(world))
        train, test = order[len(order) // 3:], order[:len(order) // 3]
    else:
        # The same observed feature must stay held out across every candidate map.
        held_out = ((np.asarray(feature_ids, dtype=np.uint64) * 2654435761) % (2 ** 32)) < (2 ** 32 // 3)
        train, test = np.flatnonzero(~held_out), np.flatnonzero(held_out)
        if len(train) < 30 or len(test) < 15:
            return {**result, "reason": "insufficient shared train/held-out features"}, None
    cv2.setRNGSeed(147)
    ok, rotation, translation, inliers = cv2.solvePnPRansac(
        world[train], pixels[train], intrinsics, None, iterationsCount=500,
        reprojectionError=3, confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 6:
        return {**result, "reason": "PnP failed"}, None
    accepted = train[inliers.ravel()]
    rotation, translation = cv2.solvePnPRefineLM(world[accepted], pixels[accepted], intrinsics,
                                               None, rotation, translation)
    projected = cv2.projectPoints(world[test], rotation, translation, intrinsics, None)[0].reshape(-1, 2)
    residuals = np.linalg.norm(projected - pixels[test], axis=1)
    height, width = size
    hull = cv2.convexHull(pixels[accepted].astype(np.float32))
    coverage = float(cv2.contourArea(hull) / (height * width))
    fraction = len(accepted) / len(train)
    test_fraction = float(np.mean(residuals <= 3))
    median = float(np.median(residuals))
    usable = len(accepted) >= 30 and fraction >= .5 and coverage >= .1 and test_fraction >= .5 and median <= 3
    pose = np.column_stack([cv2.Rodrigues(rotation)[0], translation])
    return {**result, "usable": usable, "reason": "accepted" if usable else "insufficient cross-validated geometry agreement",
            "fit_inliers": len(accepted), "fit_inlier_fraction": fraction,
            "fit_spatial_coverage": coverage, "withheld_matches": len(test),
            "withheld_fraction_within3px": test_fraction, "withheld_median_residual_px": median,
            "withheld_p90_residual_px": float(np.percentile(residuals, 90)),
            "estimated_world_to_camera": pose.tolist()}, pose


def render_points(world, colors, pose, intrinsics, size):
    height, width = size
    camera = world @ pose[:, :3].T + pose[:, 3]
    projected = camera @ intrinsics.T
    xy = np.rint(projected[:, :2] / np.maximum(projected[:, 2:], 1e-8)).astype(np.int32)
    valid = (camera[:, 2] > .001) & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    ids = np.flatnonzero(valid)
    ids = ids[np.argsort(camera[ids, 2], kind="stable")]
    addresses = xy[ids, 1] * width + xy[ids, 0]
    _, first = np.unique(addresses, return_index=True)
    image = np.zeros((height * width, 3), np.uint8)
    mask = np.zeros(height * width, bool)
    image[addresses[first]] = colors[ids[first]]
    mask[addresses[first]] = True
    return image.reshape(height, width, 3), mask.reshape(height, width)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-submaps", action="store_true",
                        help="Choose one past-view map using training inliers only; hold target features out across all maps")
    parser.add_argument("--shared-feature-holdout", action="store_true",
                        help="Use target-feature identities for the pooled control's held-out split")
    args = parser.parse_args()
    cv2.setNumThreads(4)
    geometry = np.load(args.geometry / "geometry.npz")
    metadata = json.loads((args.geometry / "run.json").read_text())
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    images, depths, confs = (geometry[name] for name in ("processed_images", "depth", "conf"))
    height, width = depths.shape[1:]
    intrinsics = np.median(geometry["intrinsics"], axis=0).astype(np.float64)
    detector = cv2.SIFT_create(nfeatures=3000)
    source_features, clouds, cloud_colors = [], [], []
    for image, depth, conf, k, pose in zip(images, depths, confs, geometry["intrinsics"], geometry["extrinsics"]):
        keys, descriptors = detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None)
        threshold = np.median(conf)
        source_features.append((keys, descriptors, depth, conf, threshold, k, pose))
        y, x = np.mgrid[:height:2, :width:2]
        valid = (conf[y, x] >= threshold) & (depth[y, x] > 0)
        pixels = np.column_stack([x[valid], y[valid]])
        clouds.append(unproject(pixels, depth[y[valid], x[valid]], k, pose))
        cloud_colors.append(image[y[valid], x[valid]])
    world, colors = np.concatenate(clouds), np.concatenate(cloud_colors)
    capture = cv2.VideoCapture(str(args.video))
    targets = [56, 60, 460, 476, 492, 508]
    assert min(targets) > metadata["latest_observed_frame"]
    rows = []
    for index in targets:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = capture.read()
        if not ok:
            raise ValueError(f"Missing declared target frame{index}")
        target = resize_like_da3(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), metadata["process_res"])
        if target.shape[:2] != (height, width):
            raise ValueError("Target and source preprocessing dimensions differ")
        keys_target, descriptors_target = detector.detectAndCompute(cv2.cvtColor(target, cv2.COLOR_RGB2GRAY), None)
        matches, view_matches = {}, [[] for _ in source_features]
        if descriptors_target is not None and len(descriptors_target) >= 2:
            for source_id, (keys, desc, depth, conf, threshold, k, pose) in enumerate(source_features):
                if desc is None:
                    continue
                for first, second in cv2.BFMatcher().knnMatch(desc, descriptors_target, k=2):
                    if first.distance >= .7 * second.distance:
                        continue
                    u, v = keys[first.queryIdx].pt
                    x, y = int(round(u)), int(round(v))
                    if not 0 <= x < width or not 0 <= y < height or conf[y, x] < threshold or depth[y, x] <= 0:
                        continue
                    point = unproject(np.array([[u, v]]), np.array([depth[y, x]]), k, pose)[0]
                    view_matches[source_id].append((first.trainIdx, first.distance, point, keys_target[first.trainIdx].pt, source_id))
                    if first.trainIdx in matches and matches[first.trainIdx][0] <= first.distance:
                        continue
                    matches[first.trainIdx] = (first.distance, point, keys_target[first.trainIdx].pt, source_id)
        points = np.array([value[1] for value in matches.values()], np.float64).reshape(-1, 3)
        pixels = np.array([value[2] for value in matches.values()], np.float64).reshape(-1, 2)
        if args.local_submaps:
            candidates = []
            for source_id, observed in enumerate(view_matches):
                # Deduplicate target features within each submap, too.
                deduplicated = {}
                for item in sorted(observed, key=lambda value: value[1]):
                    deduplicated.setdefault(item[0], item)
                values = list(deduplicated.values())
                candidate, estimate = fit_pose(
                    np.array([v[2] for v in values], np.float64).reshape(-1, 3),
                    np.array([v[3] for v in values], np.float64).reshape(-1, 2),
                    intrinsics, (height, width), feature_ids=[v[0] for v in values])
                candidates.append((candidate, estimate, source_id))
            # Never inspect held-out residuals/acceptance when choosing the map.
            row, pose, selected = max(candidates, key=lambda value: (
                value[0].get("fit_inliers", 0), value[0].get("fit_spatial_coverage", 0)))
            row = {**row, "selected_source_view": selected,
                   "candidate_diagnostics": [candidate for candidate, _, _ in candidates]}
            render_world, render_colors = clouds[selected], cloud_colors[selected]
        else:
            row, pose = fit_pose(points, pixels, intrinsics, (height, width),
                                 feature_ids=list(matches) if args.shared_feature_holdout else None)
            render_world, render_colors = world, colors
        row.update(frame=index, matched_source_views=sorted({value[3] for value in matches.values()}))
        if pose is not None:
            rendered, mask = render_points(render_world, render_colors, pose, intrinsics, (height, width))
            row["render_coverage"] = float(mask.mean())
            row["render_rgb_mae_on_covered_pixels"] = float(np.mean(np.abs(rendered[mask].astype(float) - target[mask]))) if mask.any() else None
            cv2.imwrite(str(output / f"frame-{index:03d}.jpg"), cv2.cvtColor(np.concatenate([target, rendered], axis=1), cv2.COLOR_RGB2BGR))
        rows.append(row)
    capture.release()
    report = {"scope": "frozen past-only point map localization and rendering prerequisite; not Gaussian memory improvement",
              "source_frames": metadata["input_frame_ids"], "target_frames": targets,
              "geometry_sha256": hashlib.sha256((args.geometry / "geometry.npz").read_bytes()).hexdigest(),
              "video_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
              "preprocessing": "same pinned two-stage upper-bound resize as source DA3 inputs",
              "camera_intrinsics": "median of predicted past-view intrinsics; target RGB only used to fit pose",
              "correspondence_filter": "SIFT ratio.7, past depth-confidence above each view's median; deduplicate target features",
              "acceptance": "at least30 fitted inliers, fit and withheld fractions>=.5, fitted coverage>=.1, withheld median<=3px",
              "point_count": len(world), "accepted_targets": sum(row["usable"] for row in rows), "targets": rows,
              "map_mode": "single-view local submaps; training-inlier selection; shared target-feature holdout" if args.local_submaps else "pooled global map",
              "shared_feature_holdout": args.local_submaps or args.shared_feature_holdout,
              "limitations": "Estimated depth, intrinsics and pose are not ground truth. Current observed RGB is used for localization. No generator conditioning or dynamic map updates tested."}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
