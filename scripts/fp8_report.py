#!/usr/bin/env python3
"""fp8-KV evaluation report: DINO revisit similarity, per-chunk PSNR vs the bf16 reference, memory, speed.

    python scripts/fp8_report.py --results outputs/fp8-results-h200.json --runs outputs/fp8-runs --output reproduction/fp8-kv

`--runs` holds the result directories fetched from the Modal results volume
(`modal volume get worldsync-lingbot-results <result_directory> outputs/fp8-runs`).
Diagnostic only; not a PlayWorld score. bf16 references are the `ref-bf16-*` jobs (same scene/path/seed/window).
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModel


def decode(video):
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                                "stream=width,height", "-of", "json", str(video)]))["streams"][0]
    width, height = probe["width"], probe["height"]
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = [item for path in args.results for item in json.loads(path.read_text())["results"]]
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dino = AutoModel.from_pretrained("facebook/dinov2-small").eval()

    def embed(frame):
        with torch.no_grad():
            out = dino(**processor(images=Image.fromarray(frame), return_tensors="pt"))
        return torch.nn.functional.normalize(out.last_hidden_state[:, 0], dim=-1)[0]

    rows = {}
    for result in results:
        job = result.get("id", "")
        directory = args.runs / result["result_directory"].split("/")[-1] if result.get("result_directory") else None
        if not directory or not (directory / "rollout.mp4").exists():
            rows[job] = {"status": result.get("status"), "error": result.get("error")}
            continue
        run = json.loads((directory / "run.json").read_text())
        frames = decode(directory / "rollout.mp4")
        chunk_seconds = [chunk["elapsed_s"] for chunk in run["chunks"]]
        warm_chunk_s = float(np.mean(np.diff(chunk_seconds))) if len(chunk_seconds) > 2 else None
        rows[job] = {
            "status": result.get("status"), "gpu": result.get("gpu"), "frames": int(len(frames)),
            "peak_allocated_gb": run["peak_allocated_bytes"] / 1e9,
            "job_seconds": result.get("job_seconds"), "warm_chunk_s": warm_chunk_s,
            "dino_end_vs_frame12": float(embed(frames[12]) @ embed(frames[-1])),
            "dino_end_vs_frame0": float(embed(frames[0]) @ embed(frames[-1])),
            "memory": run.get("memory"), "directory": str(directory), "_frames": frames}
    pairs = []
    for job, row in rows.items():
        match = re.match(r"(?:v2-)?fp8(?:audit)?-c(\d+)-(.+)-s(\d+)$", job)
        if not match or "_frames" not in row:
            continue
        cache, tail, seed = match.groups()
        scene_path = tail
        ref = next((other for name, other in rows.items() if name == f"ref-bf16-c46-{scene_path}-s42" and "_frames" in other), None)
        if ref is None or seed != "42":
            continue
        count = min(len(ref["_frames"]), len(row["_frames"]))
        per_chunk = [round(psnr(ref["_frames"][i * 16: (i + 1) * 16], row["_frames"][i * 16: (i + 1) * 16]), 2)
                     for i in range(count // 16)]
        row["psnr_vs_bf16_per_chunk"] = per_chunk
        row["dino_gap_vs_bf16_end"] = row["dino_end_vs_frame12"] - ref["dino_end_vs_frame12"]
        pairs.append((job, ref, row))
    sheet_rows = []
    for job, ref, row in pairs:
        sheet_rows.append((job, ref["_frames"][12], ref["_frames"][-1], row["_frames"][-1]))
    if sheet_rows:
        w, h = 320, 180
        sheet = Image.new("RGB", (w * 3, (h + 18) * len(sheet_rows)), "white")
        draw = ImageDraw.Draw(sheet)
        for r, (job, start, ref_end, fp8_end) in enumerate(sheet_rows):
            draw.text((4, r * (h + 18) + 2), f"{job}: frame12 | bf16 end | fp8 end", fill="black")
            for c, image in enumerate((start, ref_end, fp8_end)):
                sheet.paste(Image.fromarray(image).resize((w, h)), (c * w, r * (h + 18) + 18))
        sheet.save(args.output / "comparison.png")
    clean = {job: {key: value for key, value in row.items() if key != "_frames"} for job, row in rows.items()}
    (args.output / "results.json").write_text(json.dumps(clean, indent=1) + "\n")
    lines = ["| job | GPU | peak GB | warm s/chunk | DINO end vs f12 | DINO gap vs bf16 | PSNR vs bf16 (chunks 0-3) |", "|---|---|---|---|---|---|---|"]
    for job, row in sorted(clean.items()):
        if "peak_allocated_gb" not in row:
            lines.append(f"| {job} | - | - | - | {row.get('status')} {str(row.get('error'))[:60]} | - | - |")
            continue
        psnrs = row.get("psnr_vs_bf16_per_chunk")
        lines.append(f"| {job} | {row.get('gpu')} | {row['peak_allocated_gb']:.1f} | "
                     f"{row['warm_chunk_s'] and round(row['warm_chunk_s'], 2)} | {row['dino_end_vs_frame12']:.3f} | "
                     f"{round(row['dino_gap_vs_bf16_end'], 3) if 'dino_gap_vs_bf16_end' in row else '-'} | "
                     f"{psnrs[:4] if psnrs else '-'} |")
    (args.output / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
