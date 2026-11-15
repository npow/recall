"""Build docs/demo assets: horizon figure, footprint figure, side-by-side clips and strips.

Reads results from sibling worktrees (paths passed on the command line), writes docs/demo/.
No GPU and no judge calls. Selection of examples is by rule (largest paired judged differences).
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/demo"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
STOCK_C, OURS_C = (232, 120, 60), (60, 150, 232)


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def bootstrap(values, draws=10000, seed=11):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (draws, len(values))).mean(1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_horizon(tree, calls=2):
    judge = tree / "outputs/revisit-horizon-judge"
    table = {}
    for line in (tree / "reproduction/revisit-horizon-jobs.jsonl").read_text().splitlines():
        row = json.loads(line)
        run = tree / "outputs/revisit-horizon" / row["id"] / "run.json"
        video = run.with_name("rollout.mp4")
        if not run.is_file() or not video.is_file() or json.loads(run.read_text()).get("status") != "completed":
            continue
        scores, gates, cats = [], [], []
        for rep in range(1, calls + 1):
            path = judge / f"rep{rep}/episodes/{row['id']}.json"
            if path.is_file():
                data = json.loads(path.read_text())
                if not data.get("error") and (data.get("scores") or {}).get("official") is not None:
                    scores.append(float(data["scores"]["official"]))
                    gates.append((data.get("gate") or {}).get("verdict"))
                    cats.append((data.get("rubric") or {}).get("category_scores_1_to_5") or {})
        dino = judge / f"dino/{row['id']}.json"
        if len(scores) < calls:
            continue
        table[(row["task_id"], int(row["horizon_s"]), row["arm"])] = {
            "id": row["id"], "score": float(np.mean(scores)), "calls": scores, "gates": gates,
            "video": str(video), "dino": json.loads(dino.read_text())["end_vs_frame12"] if dino.is_file() else None,
            "objective": json.loads((judge / f"rep1/episodes/{row['id']}.json").read_text()).get("objective")}
    return table


def horizon_stats(table):
    report = {}
    for horizon in (0, 6, 12, 24):
        scenes = sorted({task for task, h, arm in table if h == horizon and (task, h, "stock") in table and (task, h, "memory") in table})
        if not scenes:
            report[horizon] = {"n": 0}
            continue
        stock = [table[(s, horizon, "stock")]["score"] for s in scenes]
        memory = [table[(s, horizon, "memory")]["score"] for s in scenes]
        delta = [m - s for m, s in zip(memory, stock)]
        report[horizon] = {"n": len(scenes), "scenes": scenes, "stock": bootstrap(stock), "memory": bootstrap(memory),
                           "delta": bootstrap(delta), "wins": sum(d > 1e-9 for d in delta), "losses": sum(d < -1e-9 for d in delta),
                           }
        both = [s for s in scenes if table[(s, horizon, "stock")]["dino"] is not None and table[(s, horizon, "memory")]["dino"] is not None]
        report[horizon].update({"dino_n": len(both),
                                "dino_stock": float(np.mean([table[(s, horizon, "stock")]["dino"] for s in both])),
                                "dino_memory": float(np.mean([table[(s, horizon, "memory")]["dino"] for s in both])),
                                "dino_delta": bootstrap([table[(s, horizon, "memory")]["dino"] - table[(s, horizon, "stock")]["dino"] for s in both])})
        report[horizon]["ties"] = len(scenes) - report[horizon]["wins"] - report[horizon]["losses"]
    return report


def figure_horizon(report, path, interim):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    horizons = [h for h in (0, 6, 12, 24) if report[h]["n"]]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
    fig.patch.set_facecolor("white")
    for arm, colour, label, shift in (("stock", "#e8783c", "LingBot-World2 as released", -0.25),
                                      ("memory", "#3c96e8", "with our memory", 0.25)):
        mean = [report[h][arm][0] for h in horizons]
        low = [report[h][arm][1] for h in horizons]
        high = [report[h][arm][2] for h in horizons]
        xs = [h + shift for h in horizons]
        ax.errorbar(xs, mean, yerr=[np.subtract(mean, low), np.subtract(high, mean)], color=colour, marker="o",
                    markersize=8, linewidth=2.5, capsize=5, label=label)
    for h in horizons:
        ax.annotate(f"n={report[h]['n']}", (h, 0.93), ha="center", fontsize=9, color="#555555")
    ax.set_xticks(horizons)
    ax.set_xlabel("seconds waited while looking away (plus about 6 s of turning)", fontsize=12)
    ax.set_ylabel("judged consistency score (1–5)", fontsize=12)
    ax.set_ylim(0.9, 4.0)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    title = "How long does the world hold together? 20 unseen PlayWorld scenes"
    ax.set_title(title + (" (interim)" if interim else ""), fontsize=13, loc="left")
    fig.text(0.01, 0.01, "Gemini judge, released PlayWorld GC rubric, 2 calls per video; bars: 95% bootstrap CI over scenes; scripted turn-away / wait / turn-back path",
             fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def figure_footprint(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("exact internal state\n(what we started with)", 1235, 0.780), ("8-bit internal state", 627, 0.818),
            ("our memory v2:\nkeep the frame, recompute", 6.2, 0.765)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), dpi=160)
    colours = ["#999999", "#999999", "#3c96e8"]
    names = [r[0] for r in rows]
    axes[0].barh(names, [r[1] for r in rows], color=colours)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("MB stored per remembered frame (log scale)", fontsize=10)
    for index, row in enumerate(rows):
        axes[0].text(row[1] * 1.15, index, f"{row[1]:,.0f} MB" if row[1] >= 10 else f"{row[1]} MB", va="center", fontsize=10)
    axes[0].set_xlim(1, 12000)
    axes[0].invert_yaxis()
    axes[1].barh(["stock\n(no memory)"] + names, [0.352] + [r[2] for r in rows], color=["#e8783c"] + colours)
    axes[1].set_xlabel("similarity to the start view (0–1)", fontsize=10)
    for index, value in enumerate([0.352] + [r[2] for r in rows]):
        axes[1].text(value + 0.015, index, f"{value:.2f}", va="center", fontsize=10)
    axes[1].set_xlim(0, 1)
    axes[1].invert_yaxis()
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=9)
    fig.suptitle("Compressing the model's state saves 2×. Keeping the frame and recomputing saves about 200×.", fontsize=12, x=0.02, ha="left")
    fig.text(0.01, 0.01, "One scene (GC007), one seed, scripted path, Lambda H100; similarity of the final frame to the start view. On another GPU the stock model itself returned (0.80).",
             fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


def figure_loop(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    steps = [("1  Experiment", "Memory v1 on all 171\nPlayWorld tasks\n1.80 vs stock 1.85:\nno gain", "#e8783c"),
             ("2  Audit", "Agents re-read the code\nand traces: no bug, but v1\nalso retrieves on 338 chunks\nwhere nothing is revisited", "#888888"),
             ("3  Diagnosis", "Wrong frames after\nforward-and-back moves;\n24-frame archive forgets\nafter about 6 s", "#888888"),
             ("4  Fix: memory v2", "Keep every frame, verify\nthe place before retrieving,\nrecompute state on demand\n(6 MB, not 1,235 MB)", "#3c96e8"),
             ("5  Re-test", "Slow 360° turn:\nv1 0.49 < stock 0.57\nv2 0.63\nIdentical to stock\nwhen it does not fire", "#2e9e5b")]
    fig, ax = plt.subplots(figsize=(11, 3.3), dpi=160)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.3)
    ax.axis("off")
    for index, (title, body, colour) in enumerate(steps):
        x = 0.15 + index * 2.17
        ax.add_patch(FancyBboxPatch((x, 0.55), 1.95, 2.2, boxstyle="round,pad=0.03,rounding_size=0.08", linewidth=2,
                                    edgecolor=colour, facecolor="white"))
        ax.text(x + 0.975, 2.5, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=colour)
        ax.text(x + 0.975, 1.5, body, ha="center", va="center", fontsize=8.6, color="#222222", linespacing=1.35)
        if index < len(steps) - 1:
            ax.annotate("", xy=(x + 2.17, 1.65), xytext=(x + 1.98, 1.65), arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.8))
    ax.annotate("", xy=(1.1, 0.5), xytext=(9.9, 0.5), arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.4,
                                                                       connectionstyle="arc3,rad=-0.12"))
    ax.text(5.5, 0.12, "results feed the next experiment", ha="center", fontsize=9, color="#555555")
    ax.set_title("The loop ran on its own results: experiment → audit → diagnosis → fix → re-test", fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def read_frames(path, width):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        height = int(round(frame.shape[0] * width / frame.shape[1] / 2) * 2)
        frames.append(cv2.cvtColor(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def bar(width, height, text, colour=(255, 255, 255), background=(18, 18, 18), size=18, bold=False):
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    face = font(size, bold)
    while draw.textlength(text, font=face) > width - 16 and size > 9:
        size -= 1
        face = font(size, bold)
    draw.text((8, (height - size) // 2 - 2), text, fill=colour, font=face)
    return np.asarray(image)


def wrap(draw, text, face, width):
    lines, line = [], ""
    for word in text.split():
        trial = (line + " " + word).strip()
        if draw.textlength(trial, font=face) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + [line]


def compose(left, right, panel, labels, caption, phase=None):
    width = panel * 2 + 8
    gap = np.full((left.shape[0], 8, 3), 18, np.uint8)
    top = np.concatenate([bar(panel, 30, labels[0], STOCK_C, size=17, bold=True), np.full((30, 8, 3), 18, np.uint8),
                          bar(panel, 30, labels[1], OURS_C, size=17, bold=True)], 1)
    rows = [top, np.concatenate([left, gap, right], 1), bar(width, 28, caption, (225, 225, 225), size=15)]
    if phase is not None:
        rows.append(bar(width, 24, phase, (255, 214, 102), size=14))
    return np.concatenate(rows, 0)


def write_video(frames, path, fps, crf=27):
    height, width = frames[0].shape[:2]
    process = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                                "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
                                "-preset", "slow", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    for frame in frames:
        process.stdin.write(np.ascontiguousarray(frame).tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError(f"ffmpeg failed for {path}")


def phase_text(index, total, away_s, fps=16, turn_s=3.0, lead_s=1.0):
    second = index / fps
    if second < lead_s:
        return "start view"
    if second < lead_s + turn_s:
        return "turning away"
    if second < lead_s + turn_s + away_s:
        return f"looking away ({away_s} s)"
    if second < lead_s + 2 * turn_s + away_s:
        return "turning back"
    return "back: should match the start view"


def side_by_side(stock, ours, path, labels, caption, away_s=None, panel=416, target_s=10.0, fps=16):
    left, right = read_frames(stock, panel), read_frames(ours, panel)
    count = min(len(left), len(right))
    step = max(1.0, count / (target_s * fps))
    indices = [int(i * step) for i in range(int(count / step))]
    speed = f"  ·  {step:.1f}× speed" if step > 1.05 else ""
    frames = [compose(left[i], right[i], panel, labels, caption + speed,
                      None if away_s is None else phase_text(i, count, away_s)) for i in indices]
    write_video(frames, path, fps)
    return count


def strip(stock, ours, path, labels, caption, count=6, panel=260):
    left, right = read_frames(stock, panel), read_frames(ours, panel)
    total = min(len(left), len(right))
    picks = [0] + [int(total * f) for f in (0.3, 0.5, 0.7, 0.85)] + [total - 1]
    picks = picks[:count]
    rows = []
    for name, frames, colour in ((labels[0], left, STOCK_C), (labels[1], right, OURS_C)):
        rows.append(bar(panel * count, 26, name, colour, size=15, bold=True))
        rows.append(np.concatenate([frames[i] for i in picks], 1))
    rows.append(bar(panel * count, 26, caption + "   (first frame … last frame)", (225, 225, 225), size=14))
    Image.fromarray(np.concatenate(rows, 0)).save(path, quality=88)


def title_card(width, height, lines):
    image = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(image)
    y = height // 2 - 26 * len(lines)
    for text, size, colour, bold in lines:
        face = font(size, bold)
        for part in wrap(draw, text, face, width - 80):
            draw.text(((width - draw.textlength(part, font=face)) / 2, y), part, fill=colour, font=face)
            y += int(size * 1.45)
        y += 8
    return np.asarray(image)


def hero(stock, ours, path, task, objective, scores, away_s, panel=640, fps=16):
    left, right = read_frames(stock, panel), read_frames(ours, panel)
    count = min(len(left), len(right))
    labels = ("LingBot-World2 as released", "with our memory")
    caption = f"PlayWorld scene {task}  ·  judged score {scores[0]:.1f} vs {scores[1]:.1f}"
    step = max(1.0, count / (13.0 * fps))
    indices = [int(i * step) for i in range(int(count / step))]
    body = [compose(left[i], right[i], panel, labels, caption + (f"  ·  {step:.1f}× speed" if step > 1.05 else ""),
                    phase_text(i, count, away_s)) for i in indices]
    height, width = body[0].shape[:2]
    card = title_card(width, height, [(f"PlayWorld scene {task}", 34, (255, 255, 255), True), (objective, 24, (225, 225, 225), False),
                                      ("left: LingBot-World2 as released", 22, STOCK_C, True),
                                      ("right: with our memory (training-free)", 22, OURS_C, True)])
    small = panel // 2
    def shrink(frame):
        return cv2.resize(frame, (small, int(frame.shape[0] * small / frame.shape[1])), interpolation=cv2.INTER_AREA)
    quad = np.concatenate([np.concatenate([shrink(left[12]), shrink(left[count - 1])], 1), np.full((shrink(left[0]).shape[0], 8, 3), 18, np.uint8),
                           np.concatenate([shrink(right[12]), shrink(right[count - 1])], 1)], 1)
    freeze = np.concatenate([
        np.concatenate([bar(panel, 30, labels[0], STOCK_C, size=17, bold=True), np.full((30, 8, 3), 18, np.uint8),
                        bar(panel, 30, labels[1], OURS_C, size=17, bold=True)], 1),
        np.concatenate([bar(small, 26, "start view", size=14), bar(small, 26, f"after {away_s} s away", size=14), np.full((26, 8, 3), 18, np.uint8),
                        bar(small, 26, "start view", size=14), bar(small, 26, f"after {away_s} s away", size=14)], 1),
        quad, bar(width, 28, "Does the view after turning back match the start view?", (255, 214, 102), size=16)], 0)
    extra = height - freeze.shape[0]
    if extra > 0:
        freeze = np.concatenate([np.full((extra // 2, width, 3), 18, np.uint8), freeze,
                                 np.full((extra - extra // 2, width, 3), 18, np.uint8)], 0)
    else:
        freeze = freeze[:height]
    frames = [card] * int(2.5 * fps) + body + [body[-1]] * int(0.8 * fps) + [freeze] * int(4 * fps)
    write_video(frames, path, fps, crf=25)
    Image.fromarray(freeze).save(path.with_suffix(".jpg"), quality=90)


def task_clip(stock, ours, stem, task, objective, scores, labels, panel=640, fps=16):
    """Side-by-side video of a real PlayWorld task with a title card and a start-vs-end freeze frame; also a strip."""
    left, right = read_frames(stock, panel), read_frames(ours, panel)
    count = min(len(left), len(right))
    caption = f"PlayWorld task {task}  ·  selected example"
    step = max(1.0, count / (14.0 * fps))
    indices = [int(i * step) for i in range(int(count / step))]
    body = [compose(left[i], right[i], panel, labels, caption + (f"  ·  {step:.1f}× speed" if step > 1.05 else ""), "") for i in indices]
    height, width = body[0].shape[:2]
    card = title_card(width, height, [(f"PlayWorld task {task}", 34, (255, 255, 255), True), (objective, 22, (225, 225, 225), False),
                                      (f"left: {labels[0]}", 22, STOCK_C, True), (f"right: {labels[1]}", 22, OURS_C, True)])
    small = panel // 2
    def shrink(frame):
        return cv2.resize(frame, (small, int(frame.shape[0] * small / frame.shape[1])), interpolation=cv2.INTER_AREA)
    quad = np.concatenate([np.concatenate([shrink(left[0]), shrink(left[len(left) - 1])], 1), np.full((shrink(left[0]).shape[0], 8, 3), 18, np.uint8),
                           np.concatenate([shrink(right[0]), shrink(right[len(right) - 1])], 1)], 1)
    freeze = np.concatenate([
        np.concatenate([bar(panel, 30, labels[0], STOCK_C, size=17, bold=True), np.full((30, 8, 3), 18, np.uint8),
                        bar(panel, 30, labels[1], OURS_C, size=17, bold=True)], 1),
        np.concatenate([bar(small, 26, "start view", size=14), bar(small, 26, "after returning", size=14), np.full((26, 8, 3), 18, np.uint8),
                        bar(small, 26, "start view", size=14), bar(small, 26, "after returning", size=14)], 1),
        quad, bar(width, 28, "Does the view after returning match the start view?", (255, 214, 102), size=16)], 0)
    extra = height - freeze.shape[0]
    if extra > 0:
        freeze = np.concatenate([np.full((extra // 2, width, 3), 18, np.uint8), freeze, np.full((extra - extra // 2, width, 3), 18, np.uint8)], 0)
    else:
        freeze = freeze[:height]
    write_video([card] * int(2.5 * fps) + body + [body[-1]] * int(0.8 * fps) + [freeze] * int(4 * fps), OUT / f"{stem}.mp4", fps, crf=25)
    Image.fromarray(freeze).save(OUT / f"{stem}-freeze.jpg", quality=90)
    strip(stock, ours, OUT / f"{stem}.jpg", labels, f"PlayWorld task {task}")


def memv2_cases(tree, cases):
    import csv
    rows = {row["task_id"]: row for row in csv.DictReader((tree / "reproduction/memory-v2-step2/per-case.csv").open())}
    labels = ("LingBot-World2 as released", "with our memory v2")
    for task in cases:
        row = rows[task]
        task_clip(row["stock_video"], row["memv2_video"], f"playworld-{task}", task, row["objective"],
                  (float(row["stock_mean"]), float(row["memv2_mean"])), labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memv2-cases", default="", help="comma-separated PlayWorld task ids from memory-v2 step 2 to render")
    parser.add_argument("--horizon-tree", type=Path, required=True)
    parser.add_argument("--memv2-tree", type=Path, required=True)
    parser.add_argument("--interim", action="store_true")
    parser.add_argument("--skip-media", action="store_true")
    parser.add_argument("--hero", help="scene id and horizon for the hero video, e.g. GC011:12; default = top of the rule")
    parser.add_argument("--alt-heroes", default="", help="comma-separated extra hero videos, e.g. GC034:12,GC026:12")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    table = load_horizon(args.horizon_tree)
    report = horizon_stats(table)
    figure_horizon(report, OUT / "fig1-horizon.png", bool(args.interim or len(table) < 160))
    figure_footprint(OUT / "fig2-footprint.png")
    figure_loop(OUT / "fig3-loop.png")
    pairs = []
    for (task, horizon, arm), row in table.items():
        if arm == "memory" and horizon in (12, 24) and (task, horizon, "stock") in table:
            stock = table[(task, horizon, "stock")]
            pairs.append({"task": task, "horizon": horizon, "stock": stock, "memory": row, "delta": row["score"] - stock["score"]})
    pairs.sort(key=lambda p: (-p["delta"], p["task"], p["horizon"]))
    wins, seen = [], set()
    for pair in pairs:
        if pair["task"] not in seen and pair["delta"] > 0:
            wins.append(pair)
            seen.add(pair["task"])
    losses = sorted([p for p in pairs if p["delta"] < 0], key=lambda p: (p["delta"], p["task"]))
    summary = {"interim": bool(args.interim or len(table) < 160), "episodes_scored": len(table), "horizons": {str(h): {k: v for k, v in r.items() if k != "scenes"} for h, r in report.items()},
               "pairs_h12_h24": [{"task": p["task"], "horizon": p["horizon"], "stock": p["stock"]["score"], "memory": p["memory"]["score"],
                                  "delta": p["delta"], "stock_calls": p["stock"]["calls"], "memory_calls": p["memory"]["calls"],
                                  "dino_stock": p["stock"]["dino"], "dino_memory": p["memory"]["dino"]} for p in pairs],
               "rule": "largest paired judged difference (mean of 2 Gemini calls) at 12 s or 24 s away, one entry per scene; losses = most negative"}
    pooled = {}
    for task in sorted({key[0] for key in table}):
        deltas = [table[(task, h, "memory")]["score"] - table[(task, h, "stock")]["score"] for h in (6, 12, 24)
                  if (task, h, "memory") in table and (task, h, "stock") in table]
        if deltas:
            pooled[task] = float(np.mean(deltas))
    summary["pooled_h6_h12_h24_by_scene"] = {"n": len(pooled), "delta": bootstrap(list(pooled.values())) if pooled else None}
    (OUT / "horizon-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["horizons"], indent=1))
    print("pooled", summary["pooled_h6_h12_h24_by_scene"], "episodes", len(table))
    for pair in pairs[:8] + losses[:3]:
        print(pair["task"], pair["horizon"], round(pair["stock"]["score"], 2), round(pair["memory"]["score"], 2), round(pair["delta"], 2),
              pair["stock"]["calls"], pair["memory"]["calls"], pair["stock"]["dino"], pair["memory"]["dino"])
    if args.memv2_cases:
        memv2_cases(args.memv2_tree, args.memv2_cases.split(","))
    if args.skip_media:
        return
    labels = ("LingBot-World2 as released", "with our memory")
    for kind, chosen in (("win", wins[:6]), ("loss", losses[:2])):
        for pair in chosen:
            name = f"horizon-{kind}-{pair['task']}-H{pair['horizon']}"
            caption = f"{pair['task']}  ·  {pair['horizon']} s away  ·  judged {pair['stock']['score']:.1f} vs {pair['memory']['score']:.1f}"
            strip(pair["stock"]["video"], pair["memory"]["video"], OUT / f"{name}.jpg", labels, caption)
            side_by_side(pair["stock"]["video"], pair["memory"]["video"], OUT / f"{name}.mp4", labels, caption, away_s=pair["horizon"])
    step1 = args.memv2_tree / "outputs/memv2-step1"
    labels2 = ("LingBot-World2 as released", "memory v2 (keep frame, recompute)")
    for path_name, title in (("hold", "GC007 turn / hold / return"), ("slow360", "GC007 slow 360° turn")):
        stock, ours = step1 / f"stock-{path_name}-GC007-seed42/rollout.mp4", step1 / f"v2recompute-{path_name}-GC007-seed42/rollout.mp4"
        if stock.is_file() and ours.is_file():
            strip(stock, ours, OUT / f"memv2-{path_name}-GC007.jpg", labels2, title)
            if path_name == "hold":
                side_by_side(stock, ours, OUT / "memv2-hold-GC007.mp4", labels2, title, away_s=None)
    choice = wins[0] if wins else None
    if args.hero:
        task, horizon = args.hero.split(":")
        choice = next(p for p in pairs if p["task"] == task and p["horizon"] == int(horizon))
    if choice:
        hero(choice["stock"]["video"], choice["memory"]["video"], OUT / "hero.mp4", choice["task"], choice["stock"]["objective"],
             (choice["stock"]["score"], choice["memory"]["score"]), choice["horizon"])
        for item in filter(None, args.alt_heroes.split(",")):
            task, horizon = item.split(":")
            other = next(p for p in pairs if p["task"] == task and p["horizon"] == int(horizon))
            hero(other["stock"]["video"], other["memory"]["video"], OUT / f"hero-{task}.mp4", task, other["stock"]["objective"],
                 (other["stock"]["score"], other["memory"]["score"]), other["horizon"])
        (OUT / "hero.json").write_text(json.dumps({"task": choice["task"], "horizon": choice["horizon"], "stock": choice["stock"]["score"],
                                                    "memory": choice["memory"]["score"]}) + "\n")


if __name__ == "__main__":
    main()
