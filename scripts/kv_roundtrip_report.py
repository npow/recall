"""Endpoint comparison for round-trip runs: first generated reference frame vs final frame.

Pinned DINOv2-small cosine similarity (local) plus a side-by-side image. Diagnostic only.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModel


def frame(video, index, path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", f"select=eq(n\\,{index})",
                    "-frames:v", "1", str(path)], check=True)
    return Image.open(path).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="label=run_directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").eval()

    def embed(image):
        with torch.no_grad():
            out = model(**processor(images=image, return_tensors="pt"))
        return torch.nn.functional.normalize(out.last_hidden_state[:, 0], dim=-1)[0]

    rows, report = [], {}
    reference_image = None
    for item in args.runs:
        label, directory = item.split("=", 1)
        directory = Path(directory)
        run = json.loads((directory / "run.json").read_text())
        last = run["generated_frames"] - 1
        ref = frame(directory / "rollout.mp4", 12, args.output / f"{label}-ref.png")
        end = frame(directory / "rollout.mp4", last, args.output / f"{label}-end.png")
        reference_image = reference_image or ref
        report[label] = {
            "dino_end_vs_reference_frame12": float(embed(ref) @ embed(end)),
            "dino_end_vs_stock_reference": float(embed(reference_image) @ embed(end)),
            "elapsed_s": run["elapsed_s"], "peak_allocated_gb": run["peak_allocated_bytes"] / 1e9,
            "retrieval_attention_calls": run.get("memory", {}).get("retrieval_attention_calls", 0)}
        rows.append((label, ref, end))
    width, height = 384, 216
    sheet = Image.new("RGB", (width * 2, (height + 18) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, ref, end) in enumerate(rows):
        y = i * (height + 18)
        score = report[label]["dino_end_vs_stock_reference"]
        draw.text((4, y + 2), f"{label}: left = reference (frame 12), right = final frame, DINO cos {score:.3f}", fill="black")
        sheet.paste(ref.resize((width, height)), (0, y + 18))
        sheet.paste(end.resize((width, height)), (width, y + 18))
    sheet.save(args.output / "comparison.png")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
