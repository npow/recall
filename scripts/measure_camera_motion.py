"""Measure apparent camera rotation, explicitly rejecting unreliable estimates."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import cv2
    import numpy as np
    from worldsync.lingbot_visual_pose import estimate_rotation
    from worldsync.lingbot_paths import native_frame_poses
    cv2.setNumThreads(4)
    manifest = json.loads(args.run.read_text())
    run = manifest.get("run", manifest)
    poses, _ = native_frame_poses(run["fixed_actions"], run.get("turn_degrees_per_second", 60))
    capture = cv2.VideoCapture(str(args.video))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    values = np.load(ROOT / "external/LingBot-World2/examples/03/intrinsics.npy")[0].astype(float)
    fx, fy, cx, cy = values * np.array([width / 832, height / 480, width / 832, height / 480])
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    previous = None
    previous_index = None
    pairs = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % 4 == 0:
            current = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if previous is not None:
                measured = estimate_rotation(previous, current, intrinsics)
                relative = poses[previous_index, :3, :3].T @ poses[index, :3, :3]
                commanded = float(np.degrees(np.arctan2(relative[0, 2], relative[2, 2])))
                pairs.append({"from_frame": previous_index, "to_frame": index,
                              "commanded_yaw_degrees": commanded, **measured})
            previous, previous_index = current, index
        index += 1
    capture.release()
    gains = {}
    for sign, name in ((1, "right"), (-1, "left")):
        active = [pair for pair in pairs if pair["usable"] and sign * pair["commanded_yaw_degrees"] >= 2]
        gains[name] = {"accepted_turn_pairs": len(active),
                       "apparent_to_commanded_yaw_ratio": sum(pair["apparent_yaw_degrees"] for pair in active) /
                       sum(pair["commanded_yaw_degrees"] for pair in active) if active else None}
    report = {"scope": "Post-hoc visual-motion diagnostic, not ground-truth poses or a benchmark score",
              "assumptions": "Nominal model intrinsics and a rigid pure camera rotation; parallax, wrong intrinsics and generation warps can invalidate estimates.",
              "intrinsics": intrinsics.tolist(), "sample_fps": 4, "total_pairs": len(pairs),
              "accepted_pairs": sum(pair["usable"] for pair in pairs), "turn_gains": gains, "pairs": pairs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
