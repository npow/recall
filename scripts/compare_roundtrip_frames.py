"""Compare each run's final frame with frame 0 (round trip should return to the start view).

Usage: compare_roundtrip_frames.py OUT_DIR RUN_DIR [RUN_DIR ...]
Similarity: SSIM-like global correlation on downscaled grayscale plus colour-histogram
intersection. Crude on purpose; use the side-by-side image as the real evidence.
"""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


def frame(video, index, path):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-vf",
                    f"select=eq(n\\,{index})", "-vframes", "1", str(path)], check=True)
    return Image.open(path).convert("RGB")


def frame_count(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(video)],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip())


def similarity(a, b):
    ga = np.asarray(a.convert("L").resize((96, 54)), dtype=float).ravel()
    gb = np.asarray(b.convert("L").resize((96, 54)), dtype=float).ravel()
    corr = float(np.corrcoef(ga, gb)[0, 1])
    ha = np.histogramdd(np.asarray(a.resize((96, 54))).reshape(-1, 3), bins=(8, 8, 8), range=[(0, 256)] * 3)[0]
    hb = np.histogramdd(np.asarray(b.resize((96, 54))).reshape(-1, 3), bins=(8, 8, 8), range=[(0, 256)] * 3)[0]
    hist = float(np.minimum(ha / ha.sum(), hb / hb.sum()).sum())
    return corr, hist


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    rows, results = [], {}
    for run in map(Path, sys.argv[2:]):
        video = run / "rollout.mp4"
        n = frame_count(video)
        first = frame(video, 0, out / "tmp0.png")
        last = frame(video, n - 1, out / "tmpN.png")
        corr, hist = similarity(first, last)
        info = json.loads((run / "run.json").read_text())
        results[run.name] = {"frames": n, "gray_corr": corr, "color_hist_overlap": hist,
                             "local_attn": info["local_attn_size"], "sink": info["sink_size"],
                             "peak_gb": info["peak_allocated_bytes"] / 1e9, "elapsed_s": info["elapsed_s"]}
        w = 384
        pair = Image.new("RGB", (2 * w, int(first.height * w / first.width)))
        for i, im in enumerate((first, last)):
            pair.paste(im.resize((w, pair.height)), (i * w, 0))
        rows.append((f"L{info['local_attn_size']}S{info['sink_size']}", pair))
    sheet = Image.new("RGB", (rows[0][1].width, sum(r[1].height for r in rows)))
    y = 0
    for _, im in rows:
        sheet.paste(im, (0, y)); y += im.height
    sheet.save(out / "start_vs_end.jpg", quality=90)
    (out / "similarity.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


main()
