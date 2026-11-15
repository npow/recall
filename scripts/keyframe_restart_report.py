"""Keyframe-restart report: revisit match, seam artefacts and continuity (diagnostic, not a PlayWorld score).

  python scripts/keyframe_restart_report.py --runs label=run_dir ... --output DIR

For each run directory (downloaded from the results volume): pinned DINOv2-small cosine of the final frame
vs frame 12 / frame 0, and for every restart in run.json the seam statistics: mean absolute frame difference
across the cut vs the typical frame-to-frame difference of the same video, DINO between the last frame before
the cut and the first frame after it, and gray-NCC between them. Also writes a contact sheet:
frame 12 | mid | frame before cut | frame after cut | final (non-restart runs: frame 12 | mid | final).
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModel


def load_video(path):
    reader = imageio.get_reader(path)
    frames = [np.asarray(frame) for frame in reader]
    reader.close()
    return np.stack(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="label=run_directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").eval()

    def embed(array):
        with torch.no_grad():
            out = model(**processor(images=Image.fromarray(array), return_tensors="pt"))
        return torch.nn.functional.normalize(out.last_hidden_state[:, 0], dim=-1)[0]

    def dino(a, b):
        return float(embed(a) @ embed(b))

    def gray_ncc(a, b):
        ga = np.asarray(Image.fromarray(a).convert("L").resize((112, 64)), dtype=np.float32)
        gb = np.asarray(Image.fromarray(b).convert("L").resize((112, 64)), dtype=np.float32)
        ga, gb = ga - ga.mean(), gb - gb.mean()
        denominator = float(np.linalg.norm(ga) * np.linalg.norm(gb))
        return float((ga * gb).sum() / denominator) if denominator else 0.0

    report, rows = {}, []
    for item in args.runs:
        label, directory = item.split("=", 1)
        directory = Path(directory)
        run = json.loads((directory / "run.json").read_text())
        video = load_video(directory / "rollout.mp4")
        last = len(video) - 1
        diffs = np.abs(video[1:].astype(np.float32) - video[:-1].astype(np.float32)).mean(axis=(1, 2, 3))   # diffs[i] = f[i+1] vs f[i]
        entry = {"frames": len(video), "dino_end_vs_frame12": dino(video[12], video[last]),
                 "dino_end_vs_frame0": dino(video[0], video[last]),
                 "elapsed_s": run.get("elapsed_s"), "peak_allocated_gb": run.get("peak_allocated_bytes", 0) / 1e9,
                 "termination": run.get("termination"), "restarts": []}
        panels = [("frame 12", video[12]), ("mid", video[last // 2])]
        seam_indices = []
        for restart in run.get("restarts") or []:
            cut = restart["output_frame_index"]          # first frame of the restart chunk
            seam_indices.append(cut - 1)
            mask = np.ones(len(diffs), dtype=bool)
            mask[max(cut - 2, 0):cut + 1] = False
            typical = float(np.median(diffs[mask]))
            spike = float(diffs[cut - 1])
            entry["restarts"].append({**restart, "seam_mean_abs_diff": spike, "typical_mean_abs_diff": typical,
                                      "seam_over_typical": spike / typical if typical else None,
                                      "seam_over_p95": spike / float(np.percentile(diffs[mask], 95)),
                                      "dino_before_vs_after_cut": dino(video[cut - 1], video[cut]),
                                      "ncc_before_vs_after_cut": gray_ncc(video[cut - 1], video[cut])})
            panels += [("before cut", video[cut - 1]), ("after cut", video[cut])]
        panels.append(("final", video[last]))
        entry["max_frame_diff"] = float(diffs.max())
        entry["max_frame_diff_at"] = int(diffs.argmax())
        entry["seam_frames"] = seam_indices
        report[label] = entry
        rows.append((label, panels))
    width, height = 280, 158
    columns = max(len(p) for _, p in rows)
    sheet = Image.new("RGB", (width * columns, (height + 18) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, (label, panels) in enumerate(rows):
        for c, (name, array) in enumerate(panels):
            sheet.paste(Image.fromarray(array).resize((width, height)), (c * width, r * (height + 18) + 18))
            draw.text((c * width + 3, r * (height + 18) + 3), f"{label}: {name}", fill="black")
    sheet.save(args.output / "comparison.png")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "restarts"} | {"n_restarts": len(v["restarts"])} for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    main()
