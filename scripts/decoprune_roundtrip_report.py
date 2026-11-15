"""Summarise DeCoPrune / stock round-trip runs: DINO endpoint similarity, pixel parity, pruning stats, montage.

Diagnostic on one seed and one path. Not a PlayWorld score. DINO follows scripts/compare_lingbot_roundtrip.py:
DINOv2-small, normalised CLS token, final frame vs frame 12 (last frame of chunk 0, the facade view).
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

REFERENCE_FRAME = 12


def frame(video, index, destination):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", f"select=eq(n\\,{index})",
                    "-frames:v", "1", str(destination)], check=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIR",
                        help="downloaded result directory (rollout.mp4 + run.json)")
    parser.add_argument("--parity", action="append", default=[], metavar="LABEL=BASELINE_LABEL")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for item in args.run:
        label, directory = item.split("=", 1)
        directory = Path(directory)
        runs[label] = {"dir": directory, "run": json.loads((directory / "run.json").read_text())}
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    from PIL import Image, ImageDraw
    import torch
    from transformers import AutoImageProcessor, AutoModel
    model_name = "facebook/dinov2-small"
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
    model = AutoModel.from_pretrained(model_name).eval()
    summary = {"scope": "single-seed GC007 round-trip diagnostic; no PlayWorld score",
               "feature_model": {"name": model_name, "feature": "normalised CLS token"}, "runs": {}}
    tiles = {}
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        for label, entry in runs.items():
            run, video = entry["run"], entry["dir"] / "rollout.mp4"
            last = run["generated_frames"] - 1
            middle = last // 2
            names = {"frame0": 0, "frame12": REFERENCE_FRAME, "middle": middle, "final": last}
            images = {key: Image.open(frame(video, index, scratch / f"{label}-{key}.png")).convert("RGB")
                      for key, index in names.items()}
            with torch.no_grad():
                feats = torch.nn.functional.normalize(model(**processor(
                    images=[images["frame0"], images["frame12"], images["final"]], return_tensors="pt")
                    ).last_hidden_state[:, 0], dim=-1)
            memory = run.get("memory") or {}
            summary["runs"][label] = {
                "status": run["status"], "chunks": len(run["chunks"]), "schedule": run.get("denoise_schedule", "stock"),
                "sample_shift": run["sample_shift"], "timesteps_index": run["sampler_timesteps_index"],
                "dino_final_vs_frame12": float(feats[2] @ feats[1]), "dino_final_vs_frame0": float(feats[2] @ feats[0]),
                "wall_s": run["elapsed_s"], "peak_gpu_gb": run["peak_allocated_bytes"] / 1e9,
                "pixel_sha256": [c["pixel_sha256"] for c in run["chunks"]],
                "decoprune": {k: v for k, v in memory.items() if k not in ("chunk_scores",)} if run.get("decoprune") else None,
                "chunk_scores": memory.get("chunk_scores") if run.get("decoprune") else None}
            tiles[label] = images
    for item in args.parity:
        label, baseline = item.split("=", 1)
        a, b = summary["runs"][label]["pixel_sha256"], summary["runs"][baseline]["pixel_sha256"]
        identical = 0
        for x, y in zip(a, b):
            if x != y:
                break
            identical += 1
        summary["runs"][label]["identical_leading_chunks_vs_" + baseline] = identical
    for entry in summary["runs"].values():
        entry.pop("pixel_sha256")
    (output / "decoprune-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    width, height = 416, 240
    columns = ["frame0", "frame12", "middle", "final"]
    sheet = Image.new("RGB", (width * len(columns), (height + 18) * len(tiles)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (label, images) in enumerate(tiles.items()):
        for column, key in enumerate(columns):
            sheet.paste(images[key].resize((width, height)), (column * width, row * (height + 18) + 18))
        entry = summary["runs"][label]
        draw.text((4, row * (height + 18) + 3),
                  f"{label}: DINO(final,f12)={entry['dino_final_vs_frame12']:.3f} DINO(final,f0)={entry['dino_final_vs_frame0']:.3f}"
                  f"   columns: frame 0 | frame 12 | middle | final", fill="black")
    sheet.save(output / "decoprune-comparison.png")
    print(json.dumps({label: {k: v for k, v in e.items() if k in ("dino_final_vs_frame12", "dino_final_vs_frame0", "wall_s", "peak_gpu_gb") or k.startswith("identical")}
                      for label, e in summary["runs"].items()}, indent=2))


if __name__ == "__main__":
    main()
