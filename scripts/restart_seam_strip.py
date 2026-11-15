"""Contact strip around every restart cut: (frame before | frame after) per restart, plus frame 12 and the final frame."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads((args.run / "run.json").read_text())
    reader = imageio.get_reader(args.run / "rollout.mp4")
    frames = [np.asarray(f) for f in reader]
    reader.close()
    panels = [("frame 12", frames[12])]
    for restart in run["restarts"]:
        cut = restart["output_frame_index"]
        panels += [(f"chunk {restart['global_chunk']} before", frames[cut - 1]), (f"chunk {restart['global_chunk']} after", frames[cut])]
    panels.append(("final", frames[-1]))
    w, h = 280, 158
    sheet = Image.new("RGB", (w * len(panels), h + 18), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (name, array) in enumerate(panels):
        sheet.paste(Image.fromarray(array).resize((w, h)), (i * w, 18))
        draw.text((i * w + 3, 3), name, fill="black")
    sheet.save(args.output)


if __name__ == "__main__":
    main()
