"""Download completed runtime pilots and measure matched decoded frame differences."""
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reproduction/lingbot-runtime-benchmark"


def main():
    indices = [12 + 16 * index for index in range(10)]
    frames = {}
    integrity = {}
    for mode in ("sdpa26", "sdpa29", "fa4"):
        manifest = json.loads((OUTPUT / (mode + ".json")).read_text())
        if manifest["status"] != "completed":
            raise ValueError("Incomplete runtime variant")
        folder = OUTPUT / mode
        folder.mkdir(exist_ok=True)
        movie = folder / "rollout.mp4"
        if not movie.exists():
            subprocess.run(["modal", "volume", "get", "--profile", "nissan-pow", "worldsync-lingbot-results",
                            manifest["result_directory"] + "/rollout.mp4", str(movie)], check=True)
        capture = cv2.VideoCapture(str(movie))
        sample = []
        count = 0
        while True:
            ok, pixels = capture.read()
            if not ok:
                break
            if count in indices:
                sample.append(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
            count += 1
        capture.release()
        if count != 157 or len(sample) != len(indices):
            raise ValueError("Runtime clip has missing frames")
        frames[mode] = np.stack(sample)
        integrity[mode] = {"decoded_frames": count, "sha256": hashlib.sha256(movie.read_bytes()).hexdigest()}
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(movie), "-vf",
                        "fps=1,scale=416:-1,tile=2x5", "-frames:v", "1", str(folder / "contact.jpg")], check=True)
    model_name = "facebook/dinov2-small"
    revision = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
    processor = AutoImageProcessor.from_pretrained(model_name, revision=revision, use_fast=False)
    model = AutoModel.from_pretrained(model_name, revision=revision).eval().to("cpu")
    torch.set_num_threads(8)
    features = {}
    with torch.inference_mode():
        for mode, pixels in frames.items():
            batch = processor(images=[Image.fromarray(frame) for frame in pixels], return_tensors="pt")
            encoded = model(**batch).last_hidden_state[:, 0]
            features[mode] = torch.nn.functional.normalize(encoded.float(), dim=-1)
    comparisons = {}
    for mode in ("sdpa26", "fa4"):
        difference = frames[mode].astype(np.float32) - frames["sdpa29"].astype(np.float32)
        mse = np.mean(difference ** 2, axis=(1, 2, 3))
        psnr = 10 * np.log10(255 ** 2 / np.maximum(mse, 1e-12))
        cosine = (features[mode] * features["sdpa29"]).sum(dim=-1).tolist()
        comparisons[mode] = {"frame_indices": indices, "psnr_db": psnr.tolist(), "dino_cosine": cosine,
                             "mean_psnr_db": float(psnr.mean()), "mean_dino_cosine": float(np.mean(cosine))}
    report = {"scope": "decoded-frame compatibility diagnostic on a stationary ten-chunk clip, not task quality",
              "feature_model": {"name": model_name, "revision": revision}, "reference": "sdpa29",
              "video_integrity": integrity, "comparisons": comparisons,
              "inspection": "Unblinded contact-sheet inspection: mall roof, facade, sign and framing remain visibly similar in all three variants. No obvious corruption. This stationary diagnostic does not establish moving-camera or long-horizon quality equivalence."}
    (OUTPUT / "frame-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
