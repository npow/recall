"""No-revisit degradation check: how far do LoRA / retrieval videos drift from stock on a path that never returns?

For each scene, DINOv2-small cosine between stock and each other arm at 8 evenly spaced frames, plus mean absolute pixel
difference. Diagnostic only.

  python scripts/lora_degradation.py --runs outputs/lora-eval --output reproduction/lora/degradation.json
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

PATTERN = re.compile(r"lora-eval-(?P<task>GC\d+)-(?P<arm>[a-z0-9_]+)-turn_away-seed42-\d+")


def frames(video, indices, directory):
    out = []
    for i in indices:
        path = Path(directory) / f"{video.parent.name}-{i}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", f"select=eq(n\\,{i})", "-frames:v", "1", str(path)], check=True)
        out.append(Image.open(path).convert("RGB"))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").eval()

    def embed(images):
        with torch.no_grad():
            out = model(**processor(images=images, return_tensors="pt"))
        return torch.nn.functional.normalize(out.last_hidden_state[:, 0], dim=-1)

    runs = {}
    for directory in sorted(args.runs.iterdir()):
        match = PATTERN.fullmatch(directory.name)
        if match:
            runs.setdefault(match["task"], {})[match["arm"]] = directory
    report = {}
    with tempfile.TemporaryDirectory() as scratch:
        for task, arms in runs.items():
            count = min(json.loads((d / "run.json").read_text())["generated_frames"] for d in arms.values())
            indices = [int(x) for x in np.linspace(0, count - 1, 8)]
            data = {arm: frames(d / "rollout.mp4", indices, scratch) for arm, d in arms.items()}
            ref = embed(data["stock"])
            report[task] = {}
            for arm, images in data.items():
                if arm == "stock":
                    continue
                cos = (ref * embed(images)).sum(-1).tolist()
                pixel = float(np.mean([np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)).mean()
                                       for a, b in zip(data["stock"], images)]))
                report[task][arm] = {"dino_vs_stock_mean": float(np.mean(cos)), "dino_vs_stock_min": float(np.min(cos)),
                                     "mean_abs_pixel_diff_0_255": pixel, "frames": indices}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({t: {a: {k: round(v, 3) for k, v in x.items() if k != "frames"} for a, x in d.items()} for t, d in report.items()}, indent=2))


if __name__ == "__main__":
    main()
