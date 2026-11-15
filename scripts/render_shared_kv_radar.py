"""Overlay commanded camera FOVs and real peer-retrieval events below the POVs."""
import argparse
import bisect
import json
import math
from pathlib import Path
import subprocess

import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"A": (255, 202, 101), "B": (197, 163, 255)}


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def camera_at(run, frame):
    ends, total = [], 0
    for chunk in run["chunks"]:
        total += chunk["frames"]
        ends.append(total)
    index = bisect.bisect_right(ends, frame)
    start = ends[index-1] if index else 0
    old = run["actions"][index-1]["pose_after"]["heading_degrees"] if index else 0
    new = run["actions"][index]["pose_after"]["heading_degrees"]
    heading = old+(new-old)*(frame-start+1)/run["chunks"][index]["frames"]
    trace = run["memory"]["trace"][index]
    injected = trace["attention_replaced"] and len(trace["layers_injected"]) == 40
    return heading, injected, index


def render(root, output):
    root, output = Path(root), Path(output)
    runs = {actor: json.loads((root/f"{actor}-visit.json").read_text()) for actor in COLORS}
    proof = json.loads((root/"shared-kv-proof.json").read_text())
    if not proof["peer_visual_kv_injected_into_both"]:
        raise ValueError("radar requires verified peer injection")
    # Released intrinsics are (fx, fy, cx, cy), defined for 832×480.
    import numpy as np
    intrinsics = np.load(ROOT/"external/LingBot-World2/examples/03/intrinsics.npy")[0]
    fov = math.degrees(2*math.atan(832/(2*float(intrinsics[0]))))
    capture = cv2.VideoCapture(str(root/"tiled.mp4"))
    width, height = int(capture.get(3)), int(capture.get(4))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height+320}", "-r", "16", "-i", "-", "-an", "-c:v", "libx264",
        "-crf", "23", "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
        stdin=subprocess.PIPE)
    cx, cy, radius = width//2, height+159, 117
    def xy(angle, distance):
        return cx+math.sin(math.radians(angle))*distance, cy-math.cos(math.radians(angle))*distance
    try:
        for frame in range(count):
            ok, pixels = capture.read()
            if not ok:
                raise ValueError("incomplete source video")
            canvas = Image.new("RGB", (width, height+320), "#0b0e13")
            canvas.paste(Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)), (0, 0))
            draw = ImageDraw.Draw(canvas)
            for r in (58, radius):
                draw.ellipse((cx-r, cy-r, cx+r, cy+r), outline="#344152", width=2)
            for angle in range(-180, 180, 30):
                draw.line((xy(angle, radius-5), xy(angle, radius+4)), fill="#657183", width=2)
            draw.text((cx-20, height+7), "0°", font=font(19), fill="#9aa4b2")
            # Ghost arrows mark the donor's previously observed view, not a second position.
            for actor, heading in (("A", -60), ("B", 60)):
                for r in range(15, 122, 16):
                    draw.line((xy(heading, r), xy(heading, r+7)), fill=COLORS[actor], width=2)
                x, y = xy(heading, 145)
                draw.text((x-30, y-8), f"{actor} past", font=font(18), fill=COLORS[actor])
            states = {a: camera_at(run, frame) for a, run in runs.items()}
            overlay = Image.new("RGBA", canvas.size)
            od = ImageDraw.Draw(overlay)
            for actor, (heading, injected, chunk) in states.items():
                od.pieslice((cx-radius, cy-radius, cx+radius, cy+radius),
                            heading-90-fov/2, heading-90+fov/2, fill=(*COLORS[actor], 44))
            canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(canvas)
            for actor, (heading, injected, chunk) in states.items():
                end = xy(heading, radius-5)
                draw.line(((cx, cy), end), fill=COLORS[actor], width=7 if actor == "A" else 4)
                x, y = end
                draw.ellipse((x-5, y-5, x+5, y+5), fill=COLORS[actor])
            draw.ellipse((cx-7, cy-7, cx+7, cy+7), fill="white")
            draw.text((cx-63, cy+137), "same origin", font=font(20), fill="#c8d3df")
            draw.text((32, height+30), "CAMERA FOV", font=font(29), fill="white")
            for i, actor in enumerate(COLORS):
                draw.text((32, height+83+i*46), f"{actor}: {states[actor][0]:+.0f}°", font=font(29), fill=COLORS[actor])
            draw.text((32, height+205), f"FOV ≈ {fov:.0f}° · commanded poses", font=font(22), fill="#9aa4b2")
            draw.text((32, height+246), "Dashed rays = earlier peer views", font=font(21), fill="#9aa4b2")
            right = width-520
            draw.text((right, height+30), "SHARED VISUAL KV", font=font(29), fill="#3ddc97")
            for i, actor in enumerate(COLORS):
                peer = "B" if actor == "A" else "A"
                active = states[actor][1]
                text = f"{actor} ← {peer}: 40 layers reading" if active else f"{actor}: exploring"
                draw.text((right, height+88+i*49), text, font=font(25), fill="#3ddc97" if active else "#9aa4b2")
            draw.text((right, height+218), "Pose-matched retrieval", font=font(23), fill="#c8d3df")
            draw.text((right, height+254), "Serial visits · aligned replay", font=font(21), fill="#9aa4b2")
            writer.stdin.write(canvas.tobytes())
    finally:
        capture.release()
        writer.stdin.close()
    if writer.wait():
        raise RuntimeError("radar encoding failed")
    output.with_suffix(".json").write_text(json.dumps({"horizontal_fov_degrees": fov,
        "frames": count, "source": str(root/"tiled.mp4"), "pose_source": "commanded camera path, linearly interpolated within each chunk",
        "retrieval_source": "recorded attention-replacement trace, 40 layers", "scope": "same-origin static scene, serial visits"}, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"outputs/shared-kv-revisit")
    parser.add_argument("--output", type=Path, default=ROOT/"docs/slides/media/shared-kv-radar.mp4")
    args = parser.parse_args()
    render(args.input, args.output)
