"""Build the side-by-side gallery for docs/GALLERY.md from the frozen final run `final1` (stock vs director).

    python3 scripts/build_gallery.py --source <worktree with outputs/final1 and outputs/final1-judge> [--videos 4]

Selection is by rule, not by eye: among HELD-OUT cases of the dimensions the director touches (IF, insight,
out-of-sight), the 6 largest positive and the 3 largest negative paired differences in the primary score
(single Gemini call, director minus stock), ties broken by task id, with at least one case per dimension where
that dimension has a difference of the required sign. Writes docs/gallery/*.jpg, *.mp4 and selection.json.
"""
import argparse
import json
from pathlib import Path
import subprocess
import textwrap

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/gallery"
DIMENSIONS = ("if", "insight", "outsight")
NAME = {"gc": "GC", "if": "IF", "insight": "insight", "outsight": "out-of-sight"}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def score(source, repetition, identifier):
    path = source / f"outputs/final1-judge/rep{repetition}/episodes/{identifier}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    value = (data.get("scores") or {}).get("official")
    if data.get("error") or value is None:
        return None
    return {"score": float(value), "gate": (data.get("gate") or {}).get("verdict")}


def pairs(source):
    config = json.loads((source / "reproduction/final-run-config.json").read_text())
    development = {(split, task) for split, tasks in config["development_cases_used_to_choose_the_method"].items()
                   if isinstance(tasks, list) for task in tasks}
    rows = []
    for line in (source / "reproduction/final-run-jobs.jsonl").read_text().splitlines():
        job = json.loads(line)
        if job["arm"] != "director":
            continue
        split, task = job["split"], job["task_id"]
        stock_id, director_id = f"stock-{split}-{task}-seed42", job["id"]
        stock, director = score(source, 1, stock_id), score(source, 1, director_id)
        videos = [source / f"outputs/final1/{name}/rollout.mp4" for name in (stock_id, director_id)]
        if stock is None or director is None or not all(video.is_file() for video in videos):
            continue
        rows.append({"split": split, "task_id": task, "development": (split, task) in development,
                     "stock_id": stock_id, "director_id": director_id, "stock": stock, "director": director,
                     "delta": round(director["score"] - stock["score"], 4)})
    return rows


def choose(rows, count, sign):
    held = [row for row in rows if not row["development"] and sign * row["delta"] > 0]
    held.sort(key=lambda row: (-sign * row["delta"], row["task_id"]))
    picked = held[:count]
    for dimension in DIMENSIONS:
        if any(row["split"] == dimension for row in picked):
            continue
        best = next((row for row in held if row["split"] == dimension), None)
        if best is None:
            continue
        counts = {name: sum(row["split"] == name for row in picked) for name in DIMENSIONS}
        crowded = max(counts, key=counts.get)
        victim = [row for row in picked if row["split"] == crowded][-1]
        picked[picked.index(victim)] = best
    picked.sort(key=lambda row: (-sign * row["delta"], row["task_id"]))
    return picked


def probe(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=nb_read_frames,r_frame_rate", "-of", "json", str(video)],
                         capture_output=True, text=True, check=True).stdout
    stream = json.loads(out)["streams"][0]
    numerator, denominator = stream["r_frame_rate"].split("/")
    return int(stream["nb_read_frames"]), int(numerator) / int(denominator)


def frames(video, count, width, scratch):
    total, _ = probe(video)
    indices = [round(i * (total - 1) / (count - 1)) for i in range(count)]
    images = []
    for position, index in enumerate(indices):
        target = scratch / f"{video.parent.name}-{position}.png"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-vf",
                        f"select=eq(n\\,{index}),scale={width}:-2", "-frames:v", "1", str(target)], check=True)
        images.append(Image.open(target).convert("RGB"))
    return images, total


def wrapped(draw, text, font, width):
    lines = []
    for paragraph in text.split("\n"):
        average = max(1, int(width / (draw.textlength("abcdefghij", font=font) / 10)))
        lines.extend(textwrap.wrap(paragraph, average) or [""])
    return lines


def strip(row, task, director_text, source, scratch):
    width, count = 300, 6
    stock_frames, stock_total = frames(source / f"outputs/final1/{row['stock_id']}/rollout.mp4", count, width, scratch)
    director_frames, director_total = frames(source / f"outputs/final1/{row['director_id']}/rollout.mp4", count, width, scratch)
    height = stock_frames[0].height
    small, bold = ImageFont.truetype(FONT, 15), ImageFont.truetype(BOLD, 17)
    canvas_width = width * count + 20
    probe_draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    header = [f"{row['task_id']}  ({NAME[row['split']]}, held-out, seed 42, Lambda H100)   "
              f"director minus stock = {row['delta']:+.2f} (single Gemini call each; judge noise about 0.35)"]
    objective = wrapped(probe_draw, "Task objective: " + task, small, canvas_width - 20)
    prompt = wrapped(probe_draw, "Director prompt: " + director_text, small, canvas_width - 20)[:5]
    top = 14 + 24 * len(header) + 20 * (len(objective) + len(prompt)) + 10
    canvas = Image.new("RGB", (canvas_width, top + 2 * (height + 30) + 10), "white")
    draw = ImageDraw.Draw(canvas)
    y = 10
    for line in header:
        draw.text((10, y), line, fill="black", font=bold)
        y += 24
    for line in objective + prompt:
        draw.text((10, y), line, fill=(40, 40, 40), font=small)
        y += 20
    for label, images, result, total in (("STOCK", stock_frames, row["stock"], stock_total),
                                         ("DIRECTOR", director_frames, row["director"], director_total)):
        y += 8
        draw.text((10, y), f"{label}: score {result['score']:.2f}, gate {result['gate'] or 'n/a'}, "
                           f"{total / 16:.1f} s of video", fill="black", font=bold)
        y += 22
        for position, image in enumerate(images):
            canvas.paste(image, (10 + position * width, y))
        y += height
    target = OUT / f"{row['split']}-{row['task_id']}.jpg"
    canvas.save(target, quality=82)
    return target.name


def video(row, source):
    paths = [source / f"outputs/final1/{name}/rollout.mp4" for name in (row["stock_id"], row["director_id"])]
    durations = [probe(path)[0] / 16 for path in paths]
    speed = max(1.0, max(durations) / 8.0)
    target = OUT / f"{row['split']}-{row['task_id']}.mp4"
    length = min(8.0, max(durations) / speed)
    banner = Image.new("RGB", (960, 28), "black")
    draw, font = ImageDraw.Draw(banner), ImageFont.truetype(BOLD, 17)
    draw.text((8, 4), f"STOCK  score {row['stock']['score']:.2f}", fill="white", font=font)
    draw.text((488, 4), f"DIRECTOR  score {row['director']['score']:.2f}", fill="white", font=font)
    draw.text((300, 6), f"{row['task_id']}  x{speed:.1f}", fill=(200, 200, 200), font=ImageFont.truetype(FONT, 14))
    banner_path = OUT / f".banner-{row['task_id']}.png"
    banner.save(banner_path)
    filters = [f"[{index}:v]setpts=PTS/{speed:.4f},fps=16,scale=480:268,tpad=stop_mode=clone:stop_duration=8,"
               f"trim=duration={length:.3f},setpts=PTS-STARTPTS[v{index}]" for index in (0, 1)]
    graph = ";".join(filters) + ";[v0][v1]hstack=inputs=2[h];[2:v]fps=16[b];[b][h]vstack=inputs=2:shortest=1[out]"
    for quality in (28, 32, 36):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(paths[0]), "-i", str(paths[1]), "-loop", "1",
                        "-t", f"{length:.3f}", "-i", str(banner_path), "-filter_complex",
                        graph, "-map", "[out]", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(quality),
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)], check=True)
        if target.stat().st_size <= 3_000_000:
            break
    banner_path.unlink()
    return {"file": target.name, "speed": round(speed, 2), "bytes": target.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--videos", type=int, default=4)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    rows = pairs(args.source)
    texts = json.loads((args.source / "reproduction/director-static-texts-all.json").read_text())["cases"]
    wins, losses = choose(rows, 6, +1), choose(rows, 3, -1)
    summary = {"paired_scored": len(rows), "held_out": sum(not row["development"] for row in rows), "by_dimension": {}}
    for dimension in DIMENSIONS:
        subset = [row for row in rows if row["split"] == dimension and not row["development"]]
        summary["by_dimension"][dimension] = {
            "pairs": len(subset), "wins": sum(row["delta"] > 0 for row in subset),
            "ties": sum(row["delta"] == 0 for row in subset), "losses": sum(row["delta"] < 0 for row in subset),
            "mean_delta": round(sum(row["delta"] for row in subset) / len(subset), 3) if subset else None,
            "both_at_floor": sum(row["stock"]["score"] == 1.0 and row["director"]["score"] == 1.0 for row in subset)}
    examples = []
    for kind, selected in (("win", wins), ("loss", losses)):
        for rank, row in enumerate(selected):
            from worldsync import gemini_vertex_judge
            record = gemini_vertex_judge.load_dataset_task(args.source / "external/PlayWorld", row["split"], row["task_id"])
            objective = record.get("prompt") or ""
            text = texts[f"{row['split']}:{row['task_id']}"]["static"]
            entry = {"kind": kind, "rank": rank + 1, **{key: row[key] for key in ("split", "task_id", "delta")},
                     "stock": row["stock"], "director": row["director"], "objective": objective, "director_text": text,
                     "extra_calls": {arm: [call["score"] for call in (score(args.source, rep, row[f"{arm}_id"]) for rep in (2, 3)) if call]
                                     for arm in ("stock", "director")},
                     "strip": strip(row, objective, text, args.source, args.scratch)}
            if kind == "win" and rank < args.videos:
                entry["video"] = video(row, args.source)
            examples.append(entry)
    (OUT / "selection.json").write_text(json.dumps({"rule": __doc__.split("\n\n")[2], "summary": summary,
                                                    "examples": examples}, indent=2) + "\n")
    print(json.dumps(summary, indent=1))
    for entry in examples:
        print(entry["kind"], entry["split"], entry["task_id"], entry["delta"], entry["stock"], entry["director"],
              entry["extra_calls"], entry.get("video"))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
