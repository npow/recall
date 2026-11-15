"""Side-by-side strip (stock on top, memory v2 below) for one step-2 case, plus a path comparison.

Usage: python -m scripts.memv2_pair_strip <stock dir> <memv2 dir> <out.jpg>
"""
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw


def strip(video, count=8, width=240):
    indices = np.linspace(0, len(video) - 1, count).round().astype(int)
    height = round(width * video.shape[1] / video.shape[2])
    canvas = Image.new("RGB", (width * count, height))
    for position, index in enumerate(indices):
        canvas.paste(Image.fromarray(video[index]).resize((width, height)), (position * width, 0))
    return canvas


def main():
    stock_dir, memory_dir, target = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    runs = [json.loads((d / "run.json").read_text()) for d in (stock_dir, memory_dir)]
    actions = [[a.get("action") for a in run["actions"]] for run in runs]
    hashes = [[c["pixel_sha256"] for c in run["chunks"]] for run in runs]
    same = 0
    for a, b in zip(*hashes):
        if a != b:
            break
        same += 1
    fired = [t["chunk"] for t in (runs[1].get("memory") or {}).get("trace", []) if t.get("attention_replaced")]
    print(json.dumps({"stock_actions": actions[0], "memv2_actions": actions[1], "identical_leading_chunks": same,
                      "fired_chunks": fired}))
    strips = [strip(iio.imread(d / "rollout.mp4")) for d in (stock_dir, memory_dir)]
    sheet = Image.new("RGB", (strips[0].width, strips[0].height * 2 + 28), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 1), f"stock  ({len(hashes[0])} chunks)", fill="black")
    sheet.paste(strips[0], (0, 14))
    draw.text((4, strips[0].height + 15), f"memory v2  ({len(hashes[1])} chunks; retrieval fired at {fired})", fill="black")
    sheet.paste(strips[1], (0, strips[0].height + 28))
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, quality=85)


if __name__ == "__main__":
    main()
