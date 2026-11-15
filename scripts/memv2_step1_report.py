"""Report for memory v2 fixed-path jobs: DINO of the final frame vs the start view, cost, and stock parity.

Usage: python -m scripts.memv2_step1_report <results dir with one folder per job id> --out reproduction/memory-v2/step1
"""
import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw

from worldsync.lingbot_memv2 import DinoDescriptor


def load(folder):
    run = json.loads((folder / "run.json").read_text())
    video = iio.imread(folder / "rollout.mp4") if (folder / "rollout.mp4").exists() else None
    return run, video


def strip(video, count=7, width=208):
    indices = np.linspace(0, len(video) - 1, count).round().astype(int)
    tiles = [Image.fromarray(video[i]).resize((width, round(width * video.shape[1] / video.shape[2]))) for i in indices]
    canvas = Image.new("RGB", (width * count, tiles[0].height))
    for position, tile in enumerate(tiles):
        canvas.paste(tile, (position * width, 0))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--jobs", type=Path, default=Path("reproduction/memory-v2-step1-jobs.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    jobs = [json.loads(line) for line in args.jobs.read_text().splitlines() if line.strip()]
    describe = DinoDescriptor()
    rows, videos = [], {}
    for job in jobs:
        folder = args.results / job["id"]
        if not (folder / "run.json").exists():
            rows.append({"id": job["id"], "arm": job["arm"], "path": job["path"], "task": job["task_id"], "status": "missing"})
            continue
        run, video = load(folder)
        row = {"id": job["id"], "arm": job["arm"], "path": job["path"], "task": job["task_id"],
               "status": run.get("status"), "error": run.get("error"), "chunks": len(run.get("chunks", [])),
               "gpu": (run.get("hardware") or {}).get("gpu"),
               "peak_gb": (run.get("peak_allocated_bytes") or 0) / 1e9, "elapsed_s": run.get("elapsed_s"),
               "hashes": [chunk["pixel_sha256"] for chunk in run.get("chunks", [])]}
        memory = run.get("memory") or {}
        if memory:
            trace = memory.get("trace", [])
            row.update({"fired_chunks": [t["chunk"] for t in trace if t.get("attention_replaced")],
                        "retrieved": {t["chunk"]: t["retrieved_frame_ids"] for t in trace if t.get("retrieved_frame_ids")},
                        "disk_gb_written": memory.get("disk_bytes_written", 0) / 1e9,
                        "disk_write_s": memory.get("disk_write_seconds"), "disk_read_s": memory.get("disk_read_seconds"),
                        "frames_read": memory.get("disk_frames_read"),
                        "select_read_s": memory.get("selection_and_read_seconds"),
                        "attend_overhead_s": memory.get("retrieval_attend_overhead_seconds")})
        if video is not None and row["status"] == "completed" and len(video) > 12:
            features = describe([video[0], video[12], video[-1]])
            row["dino_final_vs_frame12"] = float(features[2] @ features[1])
            row["dino_final_vs_frame0"] = float(features[2] @ features[0])
            videos[job["id"]] = video
        rows.append(row)
    stock = {(r["path"], r["task"]): r for r in rows if r["arm"] == "stock" and r.get("hashes")}
    for row in rows:
        base = stock.get((row["path"], row["task"]))
        if base and row.get("hashes"):
            row["chunks_identical_to_stock"] = sum(a == b for a, b in zip(row["hashes"], base["hashes"]))
            if base.get("elapsed_s") and row.get("elapsed_s"):
                row["extra_s_per_chunk_vs_stock"] = (row["elapsed_s"] - base["elapsed_s"]) / max(row["chunks"], 1)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "hashes"} for r in rows], indent=1) + "\n")
    lines = ["| path | scene | arm | DINO final vs frame 12 | vs frame 0 | fired chunks | same as stock | peak GB | s | disk GB | read s |",
             "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["status"] != "completed":
            lines.append(f"| {row['path']} | {row['task']} | {row['arm']} | {row['status']}: {str(row.get('error'))[:80]} |||||||||")
            continue
        fired = row.get("fired_chunks")
        lines.append(f"| {row['path']} | {row['task']} | {row['arm']} | {row.get('dino_final_vs_frame12', float('nan')):.3f} | "
                     f"{row.get('dino_final_vs_frame0', float('nan')):.3f} | {'-' if fired is None else (','.join(map(str, fired)) or 'none')} | "
                     f"{row.get('chunks_identical_to_stock', '')}/{row['chunks']} | {row['peak_gb']:.1f} | {row['elapsed_s'] or 0:.0f} | "
                     f"{row.get('disk_gb_written', 0):.0f} | {row.get('disk_read_s') or 0:.1f} |")
    (args.out / "table.md").write_text("\n".join(lines) + "\n")
    for path, task in sorted({(r["path"], r["task"]) for r in rows}):
        group = [r for r in rows if r["path"] == path and r["task"] == task and r["id"] in videos]
        if not group:
            continue
        strips = [strip(videos[r["id"]]) for r in group]
        sheet = Image.new("RGB", (strips[0].width, sum(s.height + 14 for s in strips)), "white")
        draw, y = ImageDraw.Draw(sheet), 0
        for r, image in zip(group, strips):
            draw.text((4, y + 1), f"{r['arm']}  DINO vs frame 12 = {r.get('dino_final_vs_frame12', float('nan')):.3f}", fill="black")
            sheet.paste(image, (0, y + 14))
            y += image.height + 14
        sheet.save(args.out / f"strip-{path}-{task}.jpg", quality=85)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
