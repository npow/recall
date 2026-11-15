"""Render the two-minute research presentation video from the storyboard and result files.

Usage: .venv-lingbot/bin/python scripts/build_presentation.py [--storyboard PATH] [--output PATH] [--only a,b]
Structure follows a standard research talk: problem, benchmark, baseline failures, method,
results, limitations, process, future work. Numbers are read from the source files named in
the storyboard at render time. Captions only (no voice): no text leaves this machine.
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS = 1920, 1080, 30
BG, PANEL, LINE = (13, 17, 23), (24, 30, 40), (48, 58, 74)
FG, MUTED = (236, 239, 244), (150, 160, 176)
ACCENT, OURS, WARN = (110, 168, 255), (64, 220, 160), (255, 130, 92)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf"
SECTIONS = ["Problem", "Benchmark", "Baseline failures", "Method", "Results", "Limitations", "Process", "Future work"]
_fonts = {}


def font(size, bold=True):
    key = (size, bold)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(FONT % ("-Bold" if bold else ""), size)
    return _fonts[key]


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def mix(a, b, t):
    t = min(max(t, 0.0), 1.0)
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def canvas(section=None, title=None, tag=None, title_size=60):
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    if section:
        draw.rectangle((0, 0, W, 58), fill=PANEL)
        x = 80
        for name in SECTIONS:
            active = name == section
            f = font(26, active)
            width = draw.textlength(name, font=f)
            draw.text((x, 29), name, font=f, fill=FG if active else MUTED, anchor="lm")
            if active:
                draw.rectangle((x, 52, x + width, 58), fill=ACCENT)
            x += width + 44
        if tag:
            f = font(22)
            width = draw.textlength(tag, font=f) + 36
            draw.rounded_rectangle((W - 60 - width, 10, W - 60, 48), 19, outline=ACCENT, width=2)
            draw.text((W - 60 - width / 2, 29), tag, font=f, fill=ACCENT, anchor="mm")
    if title:
        draw.text((80, 88), title, font=font(title_size), fill=FG)
    return image, draw


def caption(draw, text, alpha=1.0):
    if text:
        draw.text((W / 2, 1010), text, font=font(42), fill=mix(BG, FG, alpha), anchor="mm")


def pick(captions, t):
    for start, end, text in captions:
        if start <= t < end:
            return text, min(1.0, (t - start) / 0.35, (end - t) / 0.35)
    return "", 0.0


def wrap(draw, text, f, width):
    lines, current = [], ""
    for word in text.split():
        trial = (current + " " + word).strip()
        if draw.textlength(trial, font=f) > width and current:
            lines.append(current)
            current = word
        else:
            current = trial
    lines.append(current)
    return lines


def load_clip(path, size, step=1):
    capture = cv2.VideoCapture(str(path))
    frames, index = [], 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            frames.append(cv2.cvtColor(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        index += 1
    capture.release()
    if not frames:
        raise SystemExit(f"no frames in {path}")
    return frames


def paste(array, frame, x, y):
    array[y:y + frame.shape[0], x:x + frame.shape[1]] = frame


def lower_third(draw, x, y, w, h, text, colour, size=32):
    f = font(size)
    width = draw.textlength(text, font=f) + 34
    draw.rectangle((x, y + h - size - 24, x + width, y + h), fill=(0, 0, 0))
    draw.rectangle((x, y + h - size - 24, x + 8, y + h), fill=colour)
    draw.text((x + 20, y + h - (size + 24) / 2), text, font=f, fill=FG, anchor="lm")


def phases(actions):
    """Frame index -> phase label from the scripted action list (13 frames, then 16 per chunk)."""
    labels, seen_turn = [], False
    for index, action in enumerate(actions):
        if action == "RIGHT":
            label, seen_turn = "camera turns away", True
        elif action == "LEFT":
            label = "camera turns back"
        elif not seen_turn:
            label = "start view"
        elif "LEFT" in actions[:index]:
            label = "back at the start?"
        else:
            label = "looking away"
        labels.extend([label] * (13 if index == 0 else 16))
    return labels


class Data:
    def __init__(self, board):
        self.board = board
        self.src = {k: (Path(v) if v.startswith("/") else ROOT / v) for k, v in board["sources"].items()}
        self.horizon = json.loads(self.src["horizon_analysis"].read_text())
        self.full = json.loads(self.src["full_run"].read_text())
        self.final = json.loads(self.src["final_run"].read_text())
        self.episodes = {(r["task_id"], int(r["horizon_s"]), r["arm"]): r for r in self.horizon["per_episode"]}
        # The summary published on main (same judge files) wins where present, so the video matches the other materials.
        summary = self.src.get("horizon_summary")
        self.summary = json.loads(summary.read_text()) if summary and summary.exists() else None

    def pair(self, spec):
        scene, horizon = spec["scene"], spec["horizon"]
        out = {}
        for arm in ("stock", "memory"):
            directory = self.src["horizon_videos"] / f"rh-{arm}-{scene}-H{horizon:02d}-seed42"
            run = json.loads((directory / "run.json").read_text())
            row = self.episodes.get((scene, horizon, arm)) or {}
            out[arm] = {"video": directory / "rollout.mp4", "actions": run.get("fixed_actions") or [],
                        "score": row.get("score")}
        return out

    def horizon_rows(self):
        rows = []
        for key in sorted(self.horizon["horizons"], key=int):
            item = self.horizon["horizons"][key]
            if item["paired_scenes"]:
                row = {"seconds": int(key), "n": item["paired_scenes"], "stock": dict(item["stock"]["score"]),
                       "memory": dict(item["memory"]["score"]), "paired": dict(item["paired_score"])}
                pub = (self.summary or {}).get("horizons", {}).get(key)
                if pub and not self.summary.get("interim") and pub["n"] == row["n"]:
                    for arm in ("stock", "memory"):
                        row[arm].update(mean=pub[arm][0], ci95=pub[arm][1:])
                    row["paired"].update(mean=pub["delta"][0], ci95=pub["delta"][1:], wins=pub["wins"], ties=pub["ties"], losses=pub["losses"])
                rows.append(row)
        return rows

    def pooled(self):
        pub = (self.summary or {}).get("pooled_h6_h12_h24_by_scene")
        if pub and not self.summary.get("interim"):
            return {"mean": pub["delta"][0], "ci95": pub["delta"][1:]}
        return (self.horizon.get("pooled_h6_h12_h24") or {}).get("score")

    def horizon_final(self):
        return self.horizon["fully_scored"] >= self.horizon["jobs"]

    def baselines(self):
        a = self.full["all"]["arms"]
        b = self.final["primary_single_call"]["all_171"]
        return {"a_stock": a["stock"]["overall"], "a_memory": a["budget"]["overall"],
                "b_stock": b["stock"]["overall"], "b_director": b["director"]["overall"]}


# ---------------------------------------------------------------- scenes

def scene_title(data, seconds):
    text = data.board["text"]
    for index in range(int(seconds * FPS)):
        t = index / FPS
        image, draw = canvas()
        a = ease(t / 0.8)
        draw.text((W / 2, 380 - 30 * (1 - a)), text["title"], font=font(150), fill=mix(BG, OURS, a), anchor="mm")
        b = ease((t - 0.5) / 0.8)
        draw.text((W / 2, 535), text["subtitle"], font=font(58), fill=mix(BG, FG, b), anchor="mm")
        c = ease((t - 1.1) / 0.8)
        draw.text((W / 2, 690), text["track"], font=font(40, False), fill=mix(BG, MUTED, c), anchor="mm")
        draw.text((W / 2, 750), text["base"], font=font(40, False), fill=mix(BG, MUTED, c), anchor="mm")
        yield np.asarray(image)


def scene_problem(data, seconds):
    pair = data.pair(data.board["hero"])
    cw, ch = 1100, 635
    x, y = (W - cw) // 2, 200
    clip = load_clip(pair["stock"]["video"], (cw, ch))
    labels = phases(pair["stock"]["actions"])
    play = seconds - 4.0
    captions = [(0, play * 0.45, "A world model generates video as the camera moves."),
                (play * 0.45, play, "Look away for a few seconds, then turn back…"),
                (play, seconds, "How long can a generated world hold together?")]
    small = (760, 438)
    start, end = cv2.resize(clip[0], small), cv2.resize(clip[-1], small)
    for index in range(int(seconds * FPS)):
        t = index / FPS
        image, draw = canvas("Problem", "Generated worlds drift and forget")
        text, alpha = pick(captions, t)
        caption(draw, text, alpha)
        if t < play:
            n = min(int(t / play * len(clip)), len(clip) - 1)
            draw.text((x + cw, y - 24), labels[min(n, len(labels) - 1)] if labels else "", font=font(32, False), fill=MUTED, anchor="rm")
            array = np.array(image)
            paste(array, clip[n], x, y)
            overlay = Image.fromarray(array)
            lower_third(ImageDraw.Draw(overlay), x, y, cw, ch, "LingBot-World2 as released", MUTED)
            yield np.asarray(overlay)
        else:
            a = ease((t - play) / 0.5)
            left, right, top = 170, 990, 290
            draw.text((left, top - 34), "Start view", font=font(40), fill=mix(BG, FG, a), anchor="lm")
            draw.text((right, top - 34), "After looking away", font=font(40), fill=mix(BG, WARN, a), anchor="lm")
            draw.text((W / 2, top + small[1] + 60), "Same camera pose, a different place.", font=font(40, False), fill=mix(BG, MUTED, a), anchor="mm")
            array = np.array(image)
            paste(array, start, left, top)
            paste(array, end, right, top)
            overlay = Image.fromarray(array)
            ImageDraw.Draw(overlay).rectangle((right - 4, top - 4, right + small[0] + 4, top + small[1] + 4), outline=mix(BG, WARN, a), width=6)
            yield np.asarray(overlay)


def scene_benchmark(data, seconds):
    bench = data.board["benchmark"]
    boxes = ["start image\n+ objective", "LLM agent\nplayer", "world\nmodel", "video", "Gemini judge\nrubric, 1–5", "score"]
    widths = [270, 240, 210, 170, 290, 170]
    for index in range(int(seconds * FPS)):
        t = index / FPS
        image, draw = canvas("Benchmark", "PlayWorld: an agent plays inside the generated world", "RESEARCH RIGOUR", 54)
        x, y = 90, 210
        for n, (label, width) in enumerate(zip(boxes, widths)):
            a = ease((t - 0.3 * n) / 0.5)
            colour = ACCENT if n in (1, 2) else FG
            draw.rounded_rectangle((x, y, x + width, y + 120), 14, fill=PANEL, outline=mix(BG, colour, a), width=3)
            draw.multiline_text((x + width / 2, y + 60), label, font=font(32), fill=mix(PANEL, FG, a), anchor="mm", align="center", spacing=8)
            if n < len(boxes) - 1:
                arrow = "⇄" if n == 1 else "→"
                draw.text((x + width + 28, y + 60), arrow, font=font(44), fill=mix(BG, MUTED, a), anchor="mm")
            if n == 1:
                draw.text((x + width + 28, y + 150), "closed loop", font=font(26, False), fill=mix(BG, MUTED, a), anchor="mm")
            x += width + 56
        b = ease((t - 3.0) / 0.7)
        cx = 90
        for name, count in bench["dimensions"]:
            draw.rounded_rectangle((cx, 430, cx + 420, 570), 16, fill=PANEL, outline=mix(BG, LINE, b), width=2)
            draw.text((cx + 26, 475), name, font=font(28), fill=mix(PANEL, FG, b), anchor="lm")
            draw.text((cx + 26, 530), f"{count} tasks", font=font(30, False), fill=mix(PANEL, MUTED, b), anchor="lm")
            cx += 440
        draw.text((90, 618), "171 tasks  ·  overall score = mean of the four dimension means", font=font(34, False), fill=mix(BG, FG, b), anchor="lm")
        for n, (name, value, note) in enumerate(bench["published"]):
            c = ease((t - 4.6 - 0.4 * n) / 0.7)
            yy = 700 + n * 80
            draw.text((90, yy + 26), name, font=font(36), fill=mix(BG, FG, c), anchor="lm")
            draw.text((560, yy + 26), note, font=font(28, False), fill=mix(BG, MUTED, c), anchor="lm")
            draw.rounded_rectangle((820, yy + 4, 1620, yy + 48), 8, fill=PANEL)
            if c > 0:
                draw.rounded_rectangle((820, yy + 4, 820 + max(12, 800 * value / 2.5 * c), yy + 48), 8, fill=MUTED if n else ACCENT)
                draw.text((1650, yy + 26), f"{value * c:.2f}", font=font(38), fill=mix(BG, FG, c), anchor="lm")
        caption(draw, "Published scores: the record to beat is 2.12.", ease((t - 5.2) / 0.6))
        yield np.asarray(image)


def scene_failures(data, seconds):
    cells = data.board["failures"]
    vw, vh = 640, 369
    clips = [load_clip(c["video"], (vw, vh), c.get("step", 1)) for c in cells]
    positions = [(40, 170), (990, 170), (40, 570), (990, 570)]
    stats = data.baselines()
    for index in range(int(seconds * FPS)):
        t = index / FPS
        image, draw = canvas("Baseline failures", "Where the released model breaks", "RESEARCH RIGOUR", 56)
        for cell, (x, y) in zip(cells, positions):
            tx = x + vw + 24
            draw.text((tx, y + 6), cell["dimension"], font=font(32), fill=ACCENT, anchor="la")
            yy = y + 96
            for part in wrap(draw, "Task: " + cell["objective"], font(26, False), 250):
                draw.text((tx, yy), part, font=font(26, False), fill=MUTED, anchor="la")
                yy += 34
            yy += 14
            for part in wrap(draw, cell["failure"], font(30), 250):
                draw.text((tx, yy), part, font=font(30), fill=WARN, anchor="la")
                yy += 38
        caption(draw, f"As released: {stats['b_stock']:.2f}–{stats['a_stock']:.2f} overall; most out-of-sight episodes score 1.", ease((t - 1.0) / 0.6))
        array = np.array(image)
        for clip, (x, y) in zip(clips, positions):
            cycle = len(clip) + 24
            n = min(int(t * 16) % cycle, len(clip) - 1)
            paste(array, clip[n], x, y)
        overlay = Image.fromarray(array)
        d2 = ImageDraw.Draw(overlay)
        for cell, (x, y) in zip(cells, positions):
            d2.text((x + vw + 24, y + 50), cell["task"], font=font(26, False), fill=MUTED, anchor="la")
        yield np.asarray(overlay)


def strip_thumbs(data, count, size):
    pair = data.pair(data.board["hero"])
    frames = load_clip(pair["memory"]["video"], size)
    return [frames[int(i)] for i in np.linspace(0, len(frames) - 1, count)]


def scene_insight(data, seconds):
    count, gap, x0, y0 = 14, 126, 85, 330
    picks = strip_thumbs(data, count, (116, 67))
    dim = [(p * 0.22).astype(np.uint8) for p in picks]
    window = 4
    for index in range(int(seconds * FPS)):
        t = index / FPS
        image, draw = canvas("Method", "Key insight: the model remembers about 4.5 seconds", "NOVELTY", 56)
        head = window - 1 + ease(t / (seconds - 2.0)) * (count - window)
        last = int(round(head))
        first = last - window + 1
        draw.text((x0, y0 - 44), "generated video, left to right in time", font=font(30, False), fill=MUTED, anchor="lm")
        bx0, bx1 = x0 + first * gap - 8, x0 + last * gap + 124
        draw.rounded_rectangle((bx0, y0 - 10, bx1, y0 + 77), 10, outline=ACCENT, width=5)
        label = "rolling KV cache: 18 latent frames ≈ 4.5 s"
        lw = draw.textlength(label, font=font(36))
        lx = min(max(bx0, x0), W - 80 - lw)
        draw.text((lx, y0 + 122), label, font=font(36), fill=ACCENT, anchor="lm")
        if first > 1:
            draw.text((x0, y0 + 122), "gone", font=font(36), fill=WARN, anchor="lm")
        a = ease((t - 2.5) / 0.8)
        draw.text((W / 2, 640), "Each new frame attends only to the frames inside the window.", font=font(44, False), fill=mix(BG, FG, a), anchor="mm")
        draw.text((W / 2, 715), "Turn away for longer than that, and the place you left no longer exists for the model.",
                  font=font(36, False), fill=mix(BG, MUTED, a), anchor="mm")
        caption(draw, "Anything older than the window is forgotten.", 1.0)
        array = np.array(image)
        for n in range(last + 1):
            paste(array, picks[n] if n >= first else dim[n], x0 + n * gap, y0)
        yield array


def scene_system(data, seconds):
    rows = data.board["memory_module"]
    chips = ["training-free", "weights untouched", "window 6 + 10 + 8 = 24 frames (stock: 18)"]
    for index in range(int(seconds * FPS)):
        t = index / FPS
        image, draw = canvas("Method", "Training-free episodic memory", "NOVELTY", 60)
        a = ease(t / 0.7)
        draw.rounded_rectangle((80, 200, 640, 850), 18, fill=PANEL, outline=mix(BG, ACCENT, a), width=4)
        draw.text((110, 250), "World model", font=font(44), fill=mix(PANEL, ACCENT, a), anchor="lm")
        for n, line in enumerate(["LingBot-World2 14B", "causal video diffusion", "transformer, 40 layers", "", "KV cache per layer:", "sink 6 + recent 12", "latent frames ≈ 4.5 s"]):
            draw.text((110, 330 + n * 52), line, font=font(34, n in (4,)), fill=mix(PANEL, FG, a), anchor="lm")
        draw.rounded_rectangle((800, 200, 1840, 850), 18, outline=mix(BG, OURS, a), width=4)
        draw.text((830, 250), "Our memory module", font=font(44), fill=mix(BG, OURS, a), anchor="lm")
        for n, (name, line) in enumerate(rows):
            b = ease((t - 1.2 - 1.5 * n) / 0.7)
            y = 300 + n * 134
            draw.rounded_rectangle((830, y, 1810, y + 116), 14, fill=PANEL, outline=mix(BG, OURS, b), width=2)
            draw.text((856, y + 36), name, font=font(36), fill=mix(PANEL, OURS, b), anchor="lm")
            draw.text((856, y + 82), line, font=font(28, False), fill=mix(PANEL, FG, b), anchor="lm")
        c1 = ease((t - 1.2) / 0.7)
        draw.line((640, 358, 830, 358), fill=mix(BG, MUTED, c1), width=4)
        draw.polygon([(818, 348), (832, 358), (818, 368)], fill=mix(BG, MUTED, c1))
        draw.text((735, 330), "frames", font=font(26, False), fill=mix(BG, MUTED, c1), anchor="mm")
        c2 = ease((t - 1.2 - 1.5 * 2) / 0.7)
        draw.line((830, 626, 640, 626), fill=mix(BG, OURS, c2), width=4)
        draw.polygon([(654, 616), (640, 626), (654, 636)], fill=mix(BG, OURS, c2))
        draw.text((735, 598), "K/V", font=font(26, False), fill=mix(BG, OURS, c2), anchor="mm")
        cx = 80
        for n, chip in enumerate(chips):
            d = ease((t - 7.5 - 0.5 * n) / 0.6)
            width = draw.textlength(chip, font=font(30)) + 44
            draw.rounded_rectangle((cx, 880, cx + width, 944), 32, outline=mix(BG, OURS, d), width=3)
            draw.text((cx + width / 2, 912), chip, font=font(30), fill=mix(BG, FG, d), anchor="mm")
            cx += width + 20
        caption(draw, "New: a pose-indexed memory of the model's own earlier frames.", 1.0)
        yield np.asarray(image)


def scene_timeline(data, seconds):
    count = 11
    picks = strip_thumbs(data, count, (132, 76))
    dim = [(p * 0.28).astype(np.uint8) for p in picks]
    tiny = [cv2.resize(p, (96, 55)) for p in picks]
    s1, s2 = seconds * 0.30, seconds * 0.55
    captions = [(0, s1, "As the camera turns away, old frames leave the cache."),
                (s1, s2, "Each one is archived with its camera pose."),
                (s2, seconds - 3.2, "On the turn back, matching frames flow into attention."),
                (seconds - 3.2, seconds, "Also tested: a text director writing event captions.")]
    gap, x0, y0, ay, fy = 150, 135, 260, 520, 730
    for index in range(int(seconds * FPS)):
        t = index / FPS
        image, draw = canvas("Method", "One revisit, step by step", "NOVELTY", 60)
        shown = min(count, 1 + int(ease(t / (s1 * 0.9)) * (count - 1)))
        window = 4
        first = max(0, shown - window)
        text, alpha = pick(captions, t)
        caption(draw, text, alpha)
        bx0, bx1 = x0 + first * gap - 10, x0 + (shown - 1) * gap + 142
        draw.rounded_rectangle((bx0, y0 - 10, bx1, y0 + 86), 10, outline=ACCENT, width=4)
        label = "attention window (KV cache)"
        lw = draw.textlength(label, font=font(34))
        if bx1 - lw > x0 + 180:
            draw.text((bx1, y0 + 118), label, font=font(34), fill=ACCENT, anchor="rm")
            if first > 0:
                draw.text((x0, y0 + 118), "evicted", font=font(34, False), fill=MUTED, anchor="lm")
        else:
            draw.text((bx0, y0 + 118), label, font=font(34), fill=ACCENT, anchor="lm")
        a2 = ease((t - s1) / 0.8)
        if a2 > 0:
            draw.rounded_rectangle((x0 - 30, ay - 60, x0 + count * gap + 10, ay + 120), 16, outline=mix(BG, OURS, a2), width=3)
            draw.text((x0, ay - 28), "Episodic archive", font=font(36), fill=mix(BG, OURS, a2), anchor="lm")
            cx = x0 + 400
            for chip in ["cached K/V", "camera pose", "last 24 frames"]:
                width = draw.textlength(chip, font=font(28)) + 30
                draw.rounded_rectangle((cx, ay - 50, cx + width, ay - 6), 20, outline=mix(BG, MUTED, a2), width=2)
                draw.text((cx + width / 2, ay - 28), chip, font=font(28), fill=mix(BG, FG, a2), anchor="mm")
                cx += width + 14
        a3 = ease((t - s2) / 0.8)
        if a3 > 0:
            boxes = ["camera heads back\nto a seen place", "nearest-pose\nretrieval, up to 10",
                     "attention window:\nfirst 6 + retrieved 10 + recent 8"]
            widths = [460, 400, 640]
            bx = x0 - 30
            for n, (label2, width) in enumerate(zip(boxes, widths)):
                reveal = ease((t - s2 - 0.3 * n) / 0.5)
                draw.rounded_rectangle((bx, fy, bx + width, fy + 120), 14, outline=mix(BG, OURS if n else FG, reveal), width=3)
                draw.multiline_text((bx + width / 2, fy + 60), label2, font=font(32), fill=mix(BG, FG, reveal), anchor="mm", align="center", spacing=8)
                if n < len(boxes) - 1:
                    draw.text((bx + width + 22, fy + 60), "→", font=font(44), fill=mix(BG, MUTED, reveal), anchor="mm")
                bx += width + 44
        array = np.array(image)
        for n in range(shown):
            paste(array, picks[n] if n >= first else dim[n], x0 + n * gap, y0)
        if a2 > 0:
            kept = min(shown, 1 + int(a2 * (shown - 1))) if t < s2 else shown
            for n in range(kept):
                paste(array, tiny[n], x0 + n * gap + 18, ay + 30)
        if a3 > 0.2:
            overlay = Image.fromarray(array)
            d2 = ImageDraw.Draw(overlay)
            for n in (0, 1):
                d2.rectangle((x0 + n * gap + 14, ay + 26, x0 + n * gap + 118, ay + 89), outline=OURS, width=5)
            d2.line((x0 + 70, ay + 26, x0 + 70, y0 + 160), fill=OURS, width=4)
            d2.line((x0 + 70, y0 + 160, bx1 - 60, y0 + 160), fill=OURS, width=4)
            d2.line((bx1 - 60, y0 + 160, bx1 - 60, y0 + 96), fill=OURS, width=4)
            d2.polygon([(bx1 - 70, y0 + 106), (bx1 - 60, y0 + 88), (bx1 - 50, y0 + 106)], fill=OURS)
            array = np.asarray(overlay)
        yield array


def scene_hero(data, seconds):
    spec = data.board["hero"]
    pair = data.pair(spec)
    cw, ch = 900, 519
    y, xl, xr = 225, 40, 980
    left = load_clip(pair["stock"]["video"], (cw, ch))
    right = load_clip(pair["memory"]["video"], (cw, ch))
    labels = phases(pair["stock"]["actions"])
    freeze = 4.2
    play = seconds - freeze
    small = (590, 340)
    panels = [cv2.resize(left[0], small), cv2.resize(left[-1], small), cv2.resize(right[-1], small)]
    captions = [(0, play, "Both look away for the same time, then turn back."),
                (play, seconds, "Ours returns to the storefront it started from.")]
    score_text = ""
    if pair["stock"]["score"] is not None and pair["memory"]["score"] is not None:
        score_text = f"judge score (1–5): {pair['stock']['score']:.1f} as released, {pair['memory']['score']:.1f} with our memory"
    sub = f"PlayWorld scene {spec['scene']} · same seed, same camera path · selected example"
    for index in range(int(seconds * FPS)):
        t = index / FPS
        image, draw = canvas("Results", "Side by side: look away, come back", "RESULTS", 60)
        draw.text((80, 182), sub, font=font(28, False), fill=MUTED, anchor="lm")
        text, alpha = pick(captions, t)
        caption(draw, text, alpha)
        if t < play:
            n = min(int(t / play * len(left)), len(left) - 1)
            draw.text((W / 2, y + ch + 56), labels[min(n, len(labels) - 1)] if labels else "", font=font(38, False), fill=MUTED, anchor="mm")
            array = np.array(image)
            paste(array, left[n], xl, y)
            paste(array, right[min(n, len(right) - 1)], xr, y)
            overlay = Image.fromarray(array)
            d2 = ImageDraw.Draw(overlay)
            lower_third(d2, xl, y, cw, ch, "LingBot-World2 as released", MUTED)
            lower_third(d2, xr, y, cw, ch, "with our memory", OURS)
            yield np.asarray(overlay)
        else:
            a = ease((t - play) / 0.5)
            xs, top = [45, 665, 1285], 300
            for px, (name, colour) in zip(xs, [("Start view", FG), ("Return: as released", WARN), ("Return: with our memory", OURS)]):
                draw.text((px, top - 34), name, font=font(40), fill=mix(BG, colour, a), anchor="lm")
            draw.text((W / 2, top + small[1] + 70), score_text, font=font(40, False), fill=mix(BG, FG, a), anchor="mm")
            array = np.array(image)
            for px, panel in zip(xs, panels):
                paste(array, panel, px, top)
            overlay = Image.fromarray(array)
            d2 = ImageDraw.Draw(overlay)
            for px, colour in ((xs[1], WARN), (xs[2], OURS)):
                d2.rectangle((px - 4, top - 4, px + small[0] + 4, top + small[1] + 4), outline=mix(BG, colour, a), width=6)
            yield np.asarray(overlay)


def scene_chart(data, seconds):
    rows = data.horizon_rows()
    headline = next((r for r in rows if r["seconds"] == data.board["headline_horizon"]), rows[0])
    final = data.horizon_final()
    stats = data.baselines()
    x0, y0, cw, ch = 150, 290, 760, 450
    lo, hi = 1.0, 3.5
    max_s = max(24, max(r["seconds"] for r in rows))

    def px(seconds_away):
        return x0 + cw * seconds_away / max_s

    def py(score):
        return y0 + ch - ch * (score - lo) / (hi - lo)

    paired = headline["paired"]
    for index in range(int(seconds * FPS)):
        t = index / FPS
        image, draw = canvas("Results", "Judged score after looking away", "RESULTS · pre-registered", 60)
        for tick in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
            draw.line((x0, py(tick), x0 + cw, py(tick)), fill=LINE, width=1)
            draw.text((x0 - 20, py(tick)), f"{tick:.1f}", font=font(30, False), fill=MUTED, anchor="rm")
        for tick in sorted({0, 6, 12, 24} | {r["seconds"] for r in rows}):
            draw.text((px(tick), y0 + ch + 34), f"{tick} s", font=font(30, False), fill=MUTED, anchor="mm")
        draw.text((x0 + cw / 2, y0 + ch + 82), "seconds waited while looking away", font=font(30, False), fill=MUTED, anchor="mm")
        reveal = ease(t / 1.8)
        for arm, colour in (("stock", MUTED), ("memory", OURS)):
            points = [(px(r["seconds"]), py(r[arm]["mean"])) for r in rows]
            shown = max(1, int(math.ceil(reveal * len(points))))
            for r, point in list(zip(rows, points))[:shown]:
                low, high = r[arm]["ci95"]
                draw.line((point[0], py(min(max(low, lo), hi)), point[0], py(min(max(high, lo), hi))), fill=mix(BG, colour, 0.55), width=6)
                draw.ellipse((point[0] - 12, point[1] - 12, point[0] + 12, point[1] + 12), fill=colour)
            if shown > 1:
                draw.line(points[:shown], fill=colour, width=6)
            lx = x0 + (330 if arm == "memory" else 0)
            draw.ellipse((lx, y0 - 62, lx + 24, y0 - 38), fill=colour)
            draw.text((lx + 38, y0 - 50), "with our memory" if arm == "memory" else "as released", font=font(34), fill=colour, anchor="lm")
        a = ease((t - 0.8) / 0.7)
        rx = 1080
        pooled = data.pooled() if final else None
        big = pooled if pooled else paired
        draw.text((rx, 330), f"{big['mean']:+.2f}", font=font(170), fill=mix(BG, OURS, a), anchor="lm")
        head = "pooled over 6, 12 and 24 s away" if pooled else f"after {headline['seconds']} s away: {headline['stock']['mean']:.2f} → {headline['memory']['mean']:.2f}"
        draw.text((rx, 462), head, font=font(42), fill=mix(BG, FG, a), anchor="lm")
        draw.text((rx, 524), f"95% CI {big['ci95'][0]:+.2f} to {big['ci95'][1]:+.2f}", font=font(40, False), fill=mix(BG, FG, a), anchor="lm")
        draw.text((rx, 570), f"{headline['n']} unseen scenes · paired, same seed", font=font(34, False), fill=mix(BG, MUTED, a), anchor="lm")
        draw.text((rx, 606), "config frozen before generation", font=font(34, False), fill=mix(BG, MUTED, a), anchor="lm")
        draw.text((rx, 646), "baseline: released model · our window is 24 frames, stock 18", font=font(26, False), fill=mix(BG, MUTED, a), anchor="lm")
        for k, r in enumerate(rows):
            d = r["paired"]
            draw.text((rx, 692 + k * 40), f"{r['seconds']:>2} s:  {d['mean']:+.2f}  [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]   {d['wins']}W / {d['ties']}T / {d['losses']}L",
                      font=font(28, False), fill=mix(BG, FG, a), anchor="lm")
        if final:
            draw.text((x0, y0 + ch + 128), "Both decline with time away; the gain shrinks at 24 s (this archive holds 24 frames).",
                      font=font(26, False), fill=mix(BG, WARN, a), anchor="lm")
        else:
            draw.text((rx, 692 + len(rows) * 40 + 10), f"interim: {data.horizon['fully_scored']} of {data.horizon['jobs']} runs scored", font=font(28, False), fill=mix(BG, WARN, a), anchor="lm")
        draw.text((W / 2, 942), f"Baseline reproduced on all 171 PlayWorld tasks: {stats['b_stock']:.2f} and {stats['a_stock']:.2f} vs published 1.82",
                  font=font(34, False), fill=mix(BG, FG, ease((t - 2.2) / 0.7)), anchor="mm")
        caption(draw, "Higher judged consistency at every absence length, on 20 unseen scenes." if final else "On scenes never used in development, the judged score rises.", 1.0)
        yield np.asarray(image)


def scene_ablation(data, seconds):
    rows = data.board["ablation_rows"]
    x_label, x_bar, bar_w, top, step = 100, 1060, 620, 260, 100
    for index in range(int(seconds * FPS)):
        t = index / FPS
        image, draw = canvas("Results", "Storing less: how small can the memory be?", "RIGOUR · ablation", 58)
        draw.text((x_bar, top - 32), "revisit quality (return view vs start view, 0–1)", font=font(28, False), fill=MUTED, anchor="lm")
        for n, (label, size, value, kind) in enumerate(rows):
            y = top + n * step
            colour = {"stock": MUTED, "plain": ACCENT, "ours": OURS}[kind]
            grow = ease((t - 0.12 * n) / 0.6)
            draw.text((x_label, y + 30), label, font=font(38, kind == "ours"), fill=OURS if kind == "ours" else FG, anchor="lm")
            draw.text((x_bar - 30, y + 30), size, font=font(34, False), fill=MUTED, anchor="rm")
            draw.rounded_rectangle((x_bar, y + 6, x_bar + bar_w, y + 54), 8, fill=PANEL)
            if grow > 0:
                draw.rounded_rectangle((x_bar, y + 6, x_bar + max(12, bar_w * value * grow), y + 54), 8, fill=colour)
                draw.text((x_bar + bar_w + 24, y + 30), f"{value * grow:.2f}", font=font(38), fill=colour, anchor="lm")
        a = ease((t - 1.4) / 0.6)
        base = top + len(rows) * step
        draw.text((x_label, base + 40), "Store the frame and recompute: about 200× smaller", font=font(42), fill=mix(BG, OURS, a), anchor="lm")
        for k, note in enumerate(data.board["ablation_notes"]):
            draw.text((x_label, base + 104 + k * 44), note, font=font(28, False), fill=mix(BG, MUTED, a), anchor="lm")
        caption(draw, "Compressing the cache fails beyond 2×; recomputing works.", 1.0)
        yield np.asarray(image)


def scene_real(data, seconds):
    real = data.board["real_task"]
    small = (590, 340)
    s, v = load_clip(real["stock_video"], small), load_clip(real["ours_video"], small)
    panels = [s[0], s[-1], v[-1]]
    xs, top = [45, 665, 1285], 300
    for index in range(int(seconds * FPS)):
        image, draw = canvas("Results", "On a real PlayWorld task", "RESULTS · selected", 60)
        draw.text((80, 182), real["label"], font=font(28, False), fill=MUTED, anchor="lm")
        for px, (name, colour) in zip(xs, [("Start view", FG), ("End: as released", WARN), ("End: with our memory", OURS)]):
            draw.text((px, top - 34), name, font=font(40), fill=colour, anchor="lm")
        draw.text((W / 2, top + small[1] + 70), real["score_text"], font=font(40, False), fill=FG, anchor="mm")
        draw.text((W / 2, top + small[1] + 130), real["scope_text"], font=font(34, False), fill=MUTED, anchor="mm")
        caption(draw, "As released, a traffic light appears that was never there.", 1.0)
        array = np.array(image)
        for px, panel in zip(xs, panels):
            paste(array, panel, px, top)
        overlay = Image.fromarray(array)
        d2 = ImageDraw.Draw(overlay)
        for px, colour in ((xs[1], WARN), (xs[2], OURS)):
            d2.rectangle((px - 4, top - 4, px + small[0] + 4, top + small[1] + 4), outline=colour, width=6)
        yield np.asarray(overlay)


def scene_limits(data, seconds):
    stats = data.baselines()
    numbers = data.board["fixed_numbers"]
    loss = data.board["loss"]
    pair = data.pair(loss)
    small = (500, 288)
    a_clip, b_clip = load_clip(pair["stock"]["video"], small), load_clip(pair["memory"]["video"], small)
    bullets = [("On PlayWorld's own tasks our memory does not help.",
                "Improved design (verified retrieval), " + data.board["real_null"] + f" Full benchmark: memory {stats['a_memory']:.2f} vs {stats['a_stock']:.2f}, director {stats['b_director']:.2f} vs {stats['b_stock']:.2f}."),
               ("The model does not undo forward motion.", "A return by walking back often never happens."),
               ("Frozen scenes and unattempted actions", "are not memory problems."),
               (f"The judge is noisy: SD {numbers['judge_sd']:.2f}.", "Same video, three scores.")]
    px = 1340
    for index in range(int(seconds * FPS)):
        t = index / FPS
        image, draw = canvas("Limitations", "Where our method does not help", "INTERPRETABILITY", 60)
        for n, (head, sub) in enumerate(bullets):
            a = ease((t - 0.4 - 1.3 * n) / 0.7)
            y = 230 + n * 170
            draw.ellipse((84, y + 14, 104, y + 34), fill=mix(BG, WARN, a))
            draw.text((130, y + 24), head, font=font(38), fill=mix(BG, FG, a), anchor="lm")
            for k, part in enumerate(wrap(draw, sub, font(28, False), 1140)):
                draw.text((130, y + 78 + k * 38), part, font=font(28, False), fill=mix(BG, MUTED, a), anchor="lm")
        draw.text((px, 216), f"A loss: scene {loss['scene']}", font=font(32), fill=WARN, anchor="lm")
        draw.text((px, 560), "as released" + (f" · judge {pair['stock']['score']:.1f}" if pair["stock"]["score"] is not None else ""), font=font(28, False), fill=MUTED, anchor="lm")
        draw.text((px, 900), "with our memory" + (f" · judge {pair['memory']['score']:.1f}" if pair["memory"]["score"] is not None else ""), font=font(28, False), fill=MUTED, anchor="lm")
        caption(draw, "Failures are reported next to the wins.", 1.0)
        array = np.array(image)
        n = min(int(t / (seconds - 1.5) * len(a_clip)), len(a_clip) - 1)
        paste(array, a_clip[n], px, 250)
        paste(array, b_clip[min(n, len(b_clip) - 1)], px, 590)
        yield array


def scene_process(data, seconds):
    loop = data.board["loop"]
    chips = data.board.get("chips", ["pre-registered configs", "every run logged in the repo"])[:2]
    cx, cy, rx, ry = W // 2, 520, 640, 225
    bw, bh = 400, 132
    pts = [(cx + rx * math.cos(math.radians(-120 + 60 * n)), cy + ry * math.sin(math.radians(-120 + 60 * n))) for n in range(6)]

    def edge(c, t):
        dx, dy = t[0] - c[0], t[1] - c[1]
        k = min((bw / 2 + 10) / abs(dx) if dx else 1e9, (bh / 2 + 10) / abs(dy) if dy else 1e9)
        return (c[0] + dx * k, c[1] + dy * k)

    def arrow(draw, p0, p1, colour):
        draw.line((p0, p1), fill=colour, width=4)
        ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        tip = p1
        left = (tip[0] - 20 * math.cos(ang - 0.4), tip[1] - 20 * math.sin(ang - 0.4))
        right = (tip[0] - 20 * math.cos(ang + 0.4), tip[1] - 20 * math.sin(ang + 0.4))
        draw.polygon([tip, left, right], fill=colour)

    for index in range(int(seconds * FPS)):
        t = index / FPS
        image, draw = canvas("Process", "Autoresearch loop × past.dev", "AUTONOMY", 58)
        for n in range(6):
            a = ease((t - 0.4 - 0.5 * n) / 0.5)
            c, nxt = pts[n], pts[(n + 1) % 6]
            arrow(draw, edge(c, nxt), edge(nxt, c), mix(BG, ACCENT, ease((t - 0.7 - 0.5 * n) / 0.5)))
        for n, (name, line) in enumerate(loop):
            a = ease((t - 0.4 - 0.5 * n) / 0.5)
            x, y = pts[n]
            colour = OURS if n >= 4 else ACCENT
            draw.rounded_rectangle((x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2), 18, fill=PANEL, outline=mix(BG, colour, max(a, 0.2)), width=4)
            draw.text((x, y - 38), name, font=font(32), fill=mix(PANEL, FG, max(a, 0.3)), anchor="mm")
            for k, part in enumerate(wrap(draw, line, font(25, False), bw - 30)):
                draw.text((x, y + 2 + k * 32), part, font=font(25, False), fill=mix(PANEL, FG, a), anchor="mm")
        m = ease((t - 3.6) / 0.6)
        draw.rounded_rectangle((cx - 230, cy - 80, cx + 230, cy + 80), 18, fill=PANEL, outline=mix(BG, OURS, m), width=4)
        draw.text((cx, cy - 36), "past.dev", font=font(40), fill=mix(PANEL, OURS, m), anchor="mm")
        draw.text((cx, cy + 8), "findings memory:", font=font(28, False), fill=mix(PANEL, FG, m), anchor="mm")
        draw.text((cx, cy + 44), "100 dated, sourced findings", font=font(28, False), fill=mix(PANEL, FG, m), anchor="mm")
        r = ease((t - 4.2) / 0.6)
        arrow(draw, (cx - 90, cy - 84), (pts[0][0] + 70, pts[0][1] + bh / 2 + 8), mix(BG, OURS, r))
        draw.text((cx + 20, cy - 104), "recall past findings", font=font(26, False), fill=mix(BG, OURS, r), anchor="lm")
        arrow(draw, (pts[5][0] + bw / 2 + 8, cy), (cx - 236, cy), mix(BG, OURS, r))
        draw.text(((pts[5][0] + bw / 2 + cx - 230) / 2, cy - 44), "store the", font=font(22, False), fill=mix(BG, OURS, r), anchor="mm")
        draw.text(((pts[5][0] + bw / 2 + cx - 230) / 2, cy - 20), "result", font=font(22, False), fill=mix(BG, OURS, r), anchor="mm")
        total = sum(draw.textlength(c, font=font(30)) + 48 for c in chips) + 24 * (len(chips) - 1)
        x0 = (W - total) / 2
        for n, chip in enumerate(chips):
            d = ease((t - 4.6 - 0.3 * n) / 0.6)
            width = draw.textlength(chip, font=font(30)) + 48
            draw.rounded_rectangle((x0, 850, x0 + width, 910), 30, outline=mix(BG, ACCENT, d), width=3)
            draw.text((x0 + width / 2, 880), chip, font=font(30), fill=mix(BG, FG, d), anchor="mm")
            x0 += width + 24
        caption(draw, "AI agents ran the loop; a human set goals and budget.", 1.0)
        yield np.asarray(image)


def scene_future(data, seconds):
    text = data.board["text"]
    items = data.board["future"]
    for index in range(int(seconds * FPS)):
        t = index / FPS
        image, draw = canvas("Future work", "Future work", None, 60)
        for n, item in enumerate(items):
            a = ease((t - 0.3 - 0.6 * n) / 0.6)
            draw.ellipse((84, 258 + n * 90, 104, 278 + n * 90), fill=mix(BG, ACCENT, a))
            draw.text((130, 268 + n * 90), item, font=font(44, False), fill=mix(BG, FG, a), anchor="lm")
        b = ease((t - 2.6) / 0.7)
        draw.text((W / 2, 690), text["repo"], font=font(84), fill=mix(BG, OURS, b), anchor="mm")
        draw.text((W / 2, 800), "Built in 2 days · full tables, configs and logs in the repo", font=font(40, False), fill=mix(BG, MUTED, b), anchor="mm")
        yield np.asarray(image)


SCENES = [("title", scene_title), ("problem", scene_problem), ("benchmark", scene_benchmark), ("failures", scene_failures),
          ("insight", scene_insight), ("system", scene_system), ("timeline", scene_timeline), ("hero", scene_hero),
          ("chart", scene_chart), ("ablation", scene_ablation), ("real", scene_real), ("limits", scene_limits),
          ("process", scene_process), ("future", scene_future)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storyboard", default=str(ROOT / "reproduction/presentation-storyboard.json"))
    parser.add_argument("--output")
    parser.add_argument("--only", help="comma-separated scene names (preview)")
    args = parser.parse_args()
    board = json.loads(Path(args.storyboard).read_text())
    data = Data(board)
    output = Path(args.output) if args.output else ROOT / board["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only.split(",")) if args.only else None
    scenes = [(name, fn) for name, fn in SCENES if wanted is None or name in wanted]
    total = sum(int(board["durations"][name] * FPS) for name, _ in scenes)
    writer = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(output)], stdin=subprocess.PIPE)
    done, previous, fade = 0, None, int(0.4 * FPS)
    for name, fn in scenes:
        last = None
        for local, frame in enumerate(fn(data, board["durations"][name])):
            frame = np.array(frame, dtype=np.uint8)
            if previous is not None and local < fade:
                weight = (local + 1) / (fade + 1)
                frame = (previous.astype(np.float32) * (1 - weight) + frame.astype(np.float32) * weight).astype(np.uint8)
            last = frame
            out = frame.copy()
            out[H - 6:H, :int(W * (done + 1) / total)] = ACCENT
            writer.stdin.write(out.tobytes())
            done += 1
        previous = last
        print(name, done, flush=True)
    writer.stdin.close()
    if writer.wait():
        raise SystemExit("ffmpeg failed")
    print(json.dumps({"output": str(output), "frames": done, "seconds": done / FPS, "horizon_final": data.horizon_final()}))


if __name__ == "__main__":
    main()
