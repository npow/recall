"""Rotation-revisit report: final frame vs start frames for each method.

Reads result directories already downloaded from the Modal results volume
(`modal volume get worldsync-lingbot-results <dir> <local>`), computes pinned
DINOv2-small cosine similarity of the final frame against frame 12 (end of the
first WAIT chunk) and frame 0, and writes a table plus a side-by-side sheet.
Diagnostic only; not a PlayWorld score.
"""
import argparse
import json
from pathlib import Path
import subprocess

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
    for item in args.runs:
        label, directory = item.split("=", 1)
        directory = Path(directory)
        run = json.loads((directory / "run.json").read_text())
        video = directory / "rollout.mp4"
        last = run["generated_frames"] - 1
        first = frame(video, 0, args.output / f"{label}-f0.png")
        ref = frame(video, 12, args.output / f"{label}-f12.png")
        end = frame(video, last, args.output / f"{label}-end.png")
        mid = frame(video, last // 2, args.output / f"{label}-mid.png")
        report[label] = {
            "dino_end_vs_frame12": float(embed(ref) @ embed(end)),
            "dino_end_vs_frame0": float(embed(first) @ embed(end)),
            "dino_mid_vs_frame12": float(embed(ref) @ embed(mid)),
            "elapsed_s": run["elapsed_s"], "peak_allocated_gb": run["peak_allocated_bytes"] / 1e9,
            "generated_frames": run["generated_frames"],
            "diagnostic_controls": run.get("diagnostic_controls")}
        rows.append((label, ref, mid, end))
    width, height = 320, 180
    sheet = Image.new("RGB", (width * 3, (height + 18) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, ref, mid, end) in enumerate(rows):
        y = i * (height + 18)
        r = report[label]
        draw.text((4, y + 2), f"{label}: frame12 | mid (max turn) | final   DINO end-vs-f12 {r['dino_end_vs_frame12']:.3f}"
                  f"  end-vs-f0 {r['dino_end_vs_frame0']:.3f}", fill="black")
        for j, image in enumerate((ref, mid, end)):
            sheet.paste(image.resize((width, height)), (j * width, y + 18))
    sheet.save(args.output / "comparison.png")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: {m: round(v[m], 3) for m in ("dino_end_vs_frame12", "dino_end_vs_frame0")} for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    main()
