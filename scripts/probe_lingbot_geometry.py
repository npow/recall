"""Past-only pretrained geometry reconstruction on the local GPU.

This diagnoses prerequisites for spatial memory. It does not modify the generator,
fit Gaussian primitives, or establish physical ground-truth camera poses.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
CHECKPOINTS = {
    "DA3-SMALL": "e08cab65ca0ec38e7826075418411ab90cab4da3",
    "DA3-GIANT-1.1": "72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CHECKPOINTS, default="DA3-SMALL")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    source = ROOT / "external/Depth-Anything-3"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if revision != SOURCE_REVISION:
        raise ValueError("Unexpected geometry source revision")
    sys.path.insert(0, str(source / "src"))
    from depth_anything_3.api import DepthAnything3
    torch.set_num_threads(8)
    cv2.setNumThreads(4)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frame_ids = [12, 20, 28, 36, 44, 52]
    capture = cv2.VideoCapture(str(args.video))
    images = []
    for index in frame_ids:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = capture.read()
        if not ok:
            raise ValueError(f"Video missing source frame{index}")
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    capture.release()
    checkpoint = source / "checkpoints" / args.model
    report = {"scope": "past-only reconstruction prerequisite; no generator modification or Gaussian fitting",
              "source_revision": revision, "checkpoint": "depth-anything/" + args.model,
              "checkpoint_revision": CHECKPOINTS[args.model], "input_frame_ids": frame_ids,
              "latest_observed_frame": max(frame_ids), "process_res": 504,
              "video_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
              "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
              "cuda": torch.version.cuda, "training": False,
              "geometry_scale": "relative; no metric-depth claim", "status": "running",
              "source_patch": "Optional exporters imported lazily; unused COLMAP has no ARM wheel. Model math unchanged.",
              "checkpoint_files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in checkpoint.iterdir() if path.is_file()}}
    path = output / "run.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    started = time.monotonic()
    try:
        model = DepthAnything3.from_pretrained(str(checkpoint)).to(device="cuda").eval()
        torch.cuda.synchronize()
        report["load_s"] = time.monotonic() - started
        inference_start = time.monotonic()
        prediction = model.inference(images, process_res=504, export_dir=None)
        torch.cuda.synchronize()
        report["inference_s"] = time.monotonic() - inference_start
        arrays = {name: getattr(prediction, name) for name in
                  ("depth", "conf", "extrinsics", "intrinsics", "processed_images", "sky")
                  if getattr(prediction, name) is not None}
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("Non-finite geometry prediction")
        np.savez_compressed(output / "geometry.npz", **arrays)
        report["shapes"] = {name: list(value.shape) for name, value in arrays.items()}
        report["depth_range"] = [float(prediction.depth.min()), float(prediction.depth.max())]
        report["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - started
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
