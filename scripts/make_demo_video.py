"""Build the research-presentation video: task, dimensions, protocol, baselines, failure modes, method, results.

    .venv-lingbot/bin/python scripts/make_demo_video.py --out outputs/demo/worldsync-presentation.mp4

Video slides loop their clips for the whole slide. Scores are from reproduction/corrected-controller-validation.md
(three Gemini calls per video) and reproduction/full-run/analysis.json.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS = 1920, 1080, 16
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BG, FG, MUTED, ACCENT, GOOD, BAD, BOX = "#0f1420", "#f2f4f8", "#9aa4b8", "#6ea8fe", "#5fd38d", "#ff7a7a", "#1c2436"
CCV = ROOT / "outputs/ccv"
FOOT = "Closed-loop canonical control for interactive world models"


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def canvas(title, section=None):
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    if section:
        draw.text((110, 70), section.upper(), font=font(28, True), fill=ACCENT)
    draw.text((110, 112), title, font=font(60, True), fill=FG)
    draw.text((110, H - 70), FOOT, font=font(24), fill=MUTED)
    return image, draw


def bullets(draw, lines, y=270, x=130, size=38, gap=66):
    for line in lines:
        color = FG
        if isinstance(line, tuple):
            line, color = line
        if line.startswith("  "):
            draw.text((x + 50, y), line.strip(), font=font(size - 6), fill=MUTED if color == FG else color)
        else:
            draw.ellipse((x, y + size // 2 - 6, x + 12, y + size // 2 + 6), fill=ACCENT)
            draw.text((x + 34, y), line, font=font(size), fill=color)
        y += gap
    return y


def box(draw, xy, title, sub=None, outline=ACCENT):
    draw.rounded_rectangle(xy, radius=18, fill=BOX, outline=outline, width=3)
    x0, y0, x1, y1 = xy
    tw = draw.textlength(title, font=font(34, True))
    draw.text(((x0 + x1 - tw) / 2, y0 + 22), title, font=font(34, True), fill=FG)
    for i, line in enumerate(sub or []):
        lw = draw.textlength(line, font=font(26))
        draw.text(((x0 + x1 - lw) / 2, y0 + 76 + 36 * i), line, font=font(26), fill=MUTED)


def arrow(draw, start, end, label=None, color=MUTED):
    draw.line([start, end], fill=color, width=5)
    (x0, y0), (x1, y1) = start, end
    if x1 != x0:
        s = 1 if x1 > x0 else -1
        draw.polygon([(x1, y1), (x1 - 22 * s, y1 - 12), (x1 - 22 * s, y1 + 12)], fill=color)
    else:
        s = 1 if y1 > y0 else -1
        draw.polygon([(x1, y1), (x1 - 12, y1 - 22 * s), (x1 + 12, y1 - 22 * s)], fill=color)
    if label:
        draw.text(((x0 + x1) / 2 - draw.textlength(label, font=font(26)) / 2, min(y0, y1) - 44), label, font=font(26), fill=MUTED)


def table(draw, header, rows, xs, y, size=38, color_last=True, bold_rows=()):
    for x, cell in zip(xs, header):
        draw.text((x, y), cell, font=font(size - 4, True), fill=MUTED)
    y += size + 26
    for r, row in enumerate(rows):
        for i, (x, cell) in enumerate(zip(xs, row)):
            color = FG
            if isinstance(cell, tuple):
                cell, color = cell
            elif color_last and i == len(row) - 1 and cell.startswith("+"):
                color = GOOD
            elif color_last and i == len(row) - 1 and cell.startswith("−"):
                color = BAD
            draw.text((x, y), cell, font=font(size, r in bold_rows), fill=color)
        y += size + 24
    return y


def run(command):
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def esc(text):
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’").replace("%", "\\%")


def render_still(image, seconds, out):
    png = out.with_suffix(".png")
    image.save(png)
    run(["ffmpeg", "-y", "-loop", "1", "-t", str(seconds), "-i", str(png), "-vf", f"fps={FPS},format=yuv420p",
         "-c:v", "libx264", "-preset", "veryfast", str(out)])


def render_clips(image, clips, seconds, out):
    """Overlay looping clips on a slide background. clips: (path, x, y, w, h, speed)."""
    png = out.with_suffix(".png")
    image.save(png)
    inputs = ["-loop", "1", "-t", str(seconds), "-i", str(png)]
    graph, last = [], "0:v"
    for i, (path, x, y, w, h, speed) in enumerate(clips, start=1):
        inputs += ["-stream_loop", "-1", "-i", str(path)]
        graph.append(f"[{i}:v]setpts=PTS/{speed},scale={w}:{h}:flags=lanczos,trim=duration={seconds},setpts=PTS-STARTPTS[c{i}]")
        graph.append(f"[{last}][c{i}]overlay={x}:{y}:shortest=0:eof_action=pass[o{i}]")
        last = f"o{i}"
    graph.append(f"[{last}]fps={FPS},format=yuv420p,trim=duration={seconds}[out]")
    run(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(graph), "-map", "[out]", "-an", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "20", "-t", str(seconds), str(out)])


def grid_slide(title, section, cells, footer=None):
    """2x2 grid. cells: (path, speed, heading, caption, color)."""
    image, draw = canvas(title, section)
    w, h = 556, 310
    origins = [(200, 250), (1060, 250), (200, 655), (1060, 655)]
    clips = []
    for (path, speed, heading, caption, color), (x, y) in zip(cells, origins):
        draw.text((x, y - 42), heading, font=font(24, True), fill=color)
        draw.text((x + 4, y + h + 6), caption, font=font(22), fill=MUTED) if caption else None
        clips.append((path, x, y, w, h, speed))
    draw.rectangle((0, H - 90, W, H), fill=BG)
    return image, clips


def pair_slide(title, section, left, right, note):
    """left/right: (path, speed, heading, caption, color)."""
    image, draw = canvas(title, section)
    w, h = 832, 464
    clips = []
    for (path, speed, heading, caption, color), x in ((left, 110), (right, 978)):
        draw.text((x, 260), heading, font=font(36, True), fill=color)
        draw.text((x, 310), caption, font=font(26), fill=MUTED)
        clips.append((path, x, 370, w, h, speed))
    draw.text((110, 870), note, font=font(26), fill=MUTED)
    return image, clips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/demo/worldsync-presentation.mp4")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="slides-", dir=args.out.parent))
    parts = []

    def still(seconds, image):
        out = work / f"{len(parts):02d}.mp4"
        render_still(image, seconds, out)
        parts.append(out)

    def clips(seconds, image, items):
        out = work / f"{len(parts):02d}.mp4"
        render_clips(image, items, seconds, out)
        parts.append(out)

    # 1. Title
    image, draw = canvas("")
    draw.text((110, 360), "Closed-Loop Canonical Control", font=font(84, True), fill=FG)
    draw.text((110, 470), "for Interactive World Models", font=font(84, True), fill=FG)
    draw.text((110, 610), "Instruction following and scene memory in LingBot-World2 on the PlayWorld benchmark",
              font=font(34), fill=MUTED)
    still(6, image)

    # 2. Task
    image, draw = canvas("Task: objective-driven interactive video generation", "Problem setting")
    box(draw, (130, 330, 690, 560), "Agent player", ["Claude Haiku 4.5", "sees frames + objective", "issues camera / motion actions"])
    box(draw, (1000, 330, 1700, 560), "World model", ["LingBot-World2 14B, causal", "1 s video chunk per step", "rolling KV cache (18 latent frames)"])
    arrow(draw, (690, 400), (1000, 400), "action")
    arrow(draw, (1000, 500), (690, 500), "next video chunk")
    box(draw, (1000, 660, 1700, 860), "Rubric verifier", ["Gemini 3.1 Pro", "scores the full episode video, 1–5"], outline=MUTED)
    arrow(draw, (1350, 560), (1350, 660))
    bullets(draw, ["Input: start image + goal, no action script",
                   "Closed loop, up to ~60 s per episode"], y=680, x=130, size=30, gap=56)
    still(12, image)

    # 3. Dimensions
    image, draw = canvas("Four evaluation dimensions (171 tasks)", "PlayWorld benchmark")
    table(draw, ["Dimension", "Tasks", "What is tested"], [
        ["Geometry Consistency (GC)", "48", "Camera moves away and back: is the revisited scene the same?"],
        ["Interaction Fidelity (IF)", "50", "A character or object is told to act: does it do so correctly?"],
        ["Insight Evolution (IE)", "30", "Fixed camera, \"wait and observe\": does the visible scene evolve plausibly?"],
        ["Out-of-sight Evolution (OE)", "43", "Look away, then return: has the unseen part changed consistently?"]],
        [130, 720, 860], 300, size=32, color_last=False)
    still(14, image)

    # 4. One example per dimension
    image, items = grid_slide("One task per dimension", "PlayWorld benchmark", [
        (CCV / "C/gc-GC022/episode/rollout.mp4", 1, "GC: rotate 360° clockwise, return to the start view", None, ACCENT),
        (CCV / "C/if-IF022/episode/rollout.mp4", 1, "IF: the woman walks forward along the boardwalk", None, ACCENT),
        (CCV / "C/insight-OE020/episode/rollout.mp4", 2, "IE: wait and observe (2× speed)", None, ACCENT),
        (CCV / "C/outsight-OE055/episode/rollout.mp4", 2, "OE: move right, then back to the original view (2×)", None, ACCENT)])
    clips(14, image, items)

    # 5. Evaluation protocol
    image, draw = canvas("Evaluation protocol", "PlayWorld benchmark")
    bullets(draw, [
        "Rubric verifier: Gemini 3.1 Pro with a task-specific, weighted 1–5 rubric",
        "  video sampled at 10 fps (primary) and 0.5 fps (detail)",
        "Instruction-following gate for GC, IF and OE: Pass / Partial / Fail",
        ("  an applicable Fail scores 1, the minimum, whatever the visual quality", BAD),
        "Dimension score = mean over its tasks",
        "Overall = unweighted mean of the four dimensions",
        ("  we use gemini-3.1-pro-preview on Vertex (the released model id returns 404)", MUTED)])
    still(12, image)

    # 6. Baselines and our reproduction
    image, draw = canvas("Baselines and our reproduction", "Baselines")
    table(draw, ["Model", "GC", "IF", "IE", "OE", "Overall"], [
        ["Genie 3", "2.74", "2.40", "1.51", "1.81", "2.12"],
        ["HappyOyster", "2.54", "2.15", "1.47", "1.54", "1.92"],
        ["LingBot-World2 (published, hosted)", "2.04", "2.13", "1.95", "1.16", "1.82"],
        ["LingBot-World (v1)", "2.11", "2.23", "1.33", "1.43", "1.78"],
        ["Matrix-Game 3.0", "1.30", "1.25", "1.00", "1.00", "1.14"],
        [("LingBot-World2, our reproduction", ACCENT), ("2.41", ACCENT), ("1.84", ACCENT), ("1.68", ACCENT),
         ("1.47", ACCENT), ("1.85", ACCENT)]],
        [130, 900, 1090, 1280, 1470, 1660], 280, size=34, color_last=False, bold_rows=(5,))
    draw.text((130, 830), "Our reproduction: open 14B causal-fast checkpoint, all 171 tasks, Haiku 4.5 player; "
              "95% interval on overall 1.67–2.04", font=font(26), fill=MUTED)
    still(14, image)

    # 7. Failure modes (looping)
    image, items = grid_slide("Failure modes of the baseline", "Analysis", [
        (CCV / "L/if-IF022-attempt1/rollout.mp4", 1, "Instruction not executed (IF, score 1.0)", "\"walk forward along the boardwalk\": she walks in place", BAD),
        (CCV / "L/gc-GC022-attempt1/rollout.mp4", 1, "Trajectory abandoned (GC, 1.8)", "\"rotate 360° and return\": stops early, never returns", BAD),
        (CCV / "L/outsight-OE055-attempt1/rollout.mp4", 1, "Wrong camera motion (OE, gate fail)", "\"move right, then back\": the camera zooms instead", BAD),
        (CCV / "C/insight-OE066/episode/rollout.mp4", 2, "Object identity drift (IE, 1.0)", "piping bag morphs instead of depositing cream (2×)", BAD)])
    clips(16, image, items)

    # 8. Failure statistics
    image, draw = canvas("Where the baseline loses points (171-task run)", "Analysis")
    bullets(draw, [
        "36% of IF tasks fail the instruction gate (18 of 50) and are floored at 1",
        "19 GC/OE tasks require a revisit that never happens (8 GC, 11 OE)",
        "The free-form player ends moving episodes after 5–10 s of a 60 s budget",
        "Out-of-sight is the weakest dimension (published LingBot-World2: 1.16 of 5)",
        ("Most lost points come from control, not from the generator's image quality", ACCENT)])
    still(10, image)

    # 9. Method
    image, draw = canvas("Method: closed-loop canonical control", "Approach")
    box(draw, (110, 290, 520, 470), "Task objective", ["text goal + action DSL", "e.g. \"→×16\", \"w×8 → s×8\""], outline=MUTED)
    box(draw, (620, 290, 1080, 470), "Reference timeline", ["quarter-second schedule", "independent of chunking"])
    box(draw, (1180, 290, 1800, 470), "Canonical camera plan", ["absolute target poses", "explicit return-to-start bindings"])
    box(draw, (1180, 570, 1800, 750), "Fixed-scale conditioning", ["Plücker camera embedding", "no per-chunk renormalization"])
    box(draw, (620, 570, 1080, 750), "LingBot-World2 14B", ["frozen weights", "60 × 1 s chunks"], outline=MUTED)
    box(draw, (110, 570, 520, 750), "Hold service", ["released reference prompt", "hold / advance per chunk"])
    arrow(draw, (520, 380), (620, 380))
    arrow(draw, (1080, 380), (1180, 380))
    arrow(draw, (1490, 470), (1490, 570))
    arrow(draw, (1180, 660), (1080, 660))
    arrow(draw, (620, 660), (520, 660), "frames")
    draw.text((110, 800), "Baseline: a free-form player picks one discrete action per chunk, relative poses are renormalized "
              "each chunk,", font=font(26), fill=MUTED)
    draw.text((110, 840), "and the player may stop at any time. Ours fixes the trajectory in advance and lets the player "
              "only gate its progress.", font=font(26), fill=MUTED)
    still(16, image)

    # 10. Setup
    image, draw = canvas("Experimental setup", "Experiments")
    bullets(draw, [
        "12 tasks, 3 per dimension, chosen by hash of the task id (seed 42)",
        "Paired: baseline and ours on the same task, seed and GPU",
        "Baseline = the frozen configuration of our 171-task reproduction",
        "Every video scored by 3 independent verifier calls; we report the mean",
        "Same model weights in both arms: only control and conditioning differ"])
    still(8, image)

    # 11. Results
    image, draw = canvas("Results: paired comparison on 12 tasks", "Experiments")
    table(draw, ["Dimension", "Baseline", "Ours", "Δ"], [
        ["Geometry Consistency", "2.53", "2.96", "+0.42"], ["Interaction Fidelity", "1.33", "1.73", "+0.39"],
        ["Insight Evolution", "1.22", "1.20", "−0.02"], ["Out-of-sight Evolution", "1.00", "1.00", "0.00"],
        ["Overall", "1.52", "1.72", "+0.20"], ["Instruction gate: Pass", "5 / 36", "9 / 36", "+4"],
        ["Instruction gate: Fail", "9 / 36", "5 / 36", ("−4", GOOD)]],
        [130, 900, 1160, 1420], 270, size=36, bold_rows=(4,))
    still(12, image)

    # 12-13. Qualitative pairs (looping)
    note = "Mean of 3 Gemini 3.1 Pro verifier calls, released PlayWorld rubric, 1–5. Same seed and GPU."
    image, items = pair_slide("GC022: \"Rotate 360° clockwise, then return to the original view\"", "Qualitative",
        (CCV / "L/gc-GC022-attempt1/rollout.mp4", 1, "Baseline: 1.8", "stops partway through the turn; never returns", BAD),
        (CCV / "C/gc-GC022/episode/rollout.mp4", 1, "Ours: 4.6", "full 360° turn, then returns to the start view", GOOD), note)
    clips(14, image, items)
    image, items = pair_slide("IF022: \"The woman walks forward along the wooden boardwalk\"", "Qualitative",
        (CCV / "L/if-IF022-attempt1/rollout.mp4", 1, "Baseline: 1.0 (gate fail)", "she walks in place; the scene barely moves", BAD),
        (CCV / "C/if-IF022/episode/rollout.mp4", 1, "Ours: 2.1", "she walks along the boardwalk, then stops", GOOD), note)
    clips(12, image, items)

    # 14. Limitations
    image, draw = canvas("Limitations and ongoing work", "Discussion")
    bullets(draw, [
        "12 tasks and one seed: the overall gain (+0.20) is within noise; IF and gate gains are the clearest signal",
        "A replication on different GPUs disagrees on GC: per-task variance is large",
        "Out-of-sight evolution stays at the floor: memory beyond the 18-frame cache is unsolved",
        "Ongoing: the paired IF comparison on all 49 supported tasks"], size=32, gap=72)
    still(10, image)

    listing = work / "list.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in parts))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(args.out)])
    print(args.out)


if __name__ == "__main__":
    main()
