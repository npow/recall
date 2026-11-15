"""Table for the memory-storage ablation: storage bytes vs revisit similarity (DINO; diagnostic only).

Reads outputs/memstore/<id>/{run.json,rollout.mp4}, computes DINOv2-small cosine of the final frame against
frame 12 and frame 0, pulls storage bytes from the run's memory metadata, and writes a table, a JSON file and
one comparison sheet per (scene, path).
"""
import argparse
import json
from pathlib import Path
import subprocess

import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[1]


def frame(video, index):
    data = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select=eq(n\\,{index})",
                           "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                          check=True, capture_output=True).stdout
    import io
    return Image.open(io.BytesIO(data)).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, default=ROOT / "reproduction/memory-storage-jobs.jsonl")
    parser.add_argument("--results", type=Path, default=ROOT / "outputs/memstore")
    parser.add_argument("--output", type=Path, default=ROOT / "reproduction/memory-storage-ablation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").eval()

    def embed(image):
        with torch.no_grad():
            out = model(**processor(images=image, return_tensors="pt"))
        return torch.nn.functional.normalize(out.last_hidden_state[:, 0], dim=-1)[0]

    rows, sheets = [], {}
    for line in args.jobs.read_text().splitlines():
        job = json.loads(line)
        directory = args.results / job["id"]
        entry = {"id": job["id"], "variant": job["arm"], "task": job["task_id"], "path": job["path"],
                 "seed": job["seed"], "status": "missing"}
        run_path = directory / "run.json"
        if run_path.exists():
            run = json.loads(run_path.read_text())
            entry["status"] = run.get("status")
            video = directory / "rollout.mp4"
            if run.get("status") == "completed" and video.exists():
                last = run["generated_frames"] - 1
                first, ref, end, mid = frame(video, 0), frame(video, 12), frame(video, last), frame(video, last // 2)
                memory = run.get("memory") or {}
                trace = memory.get("trace") or []
                entry.update({
                    "dino_end_vs_frame12": float(embed(ref) @ embed(end)),
                    "dino_end_vs_frame0": float(embed(first) @ embed(end)),
                    "bytes_per_frame": memory.get("stored_bytes_per_archived_frame"),
                    "exact_bytes_per_frame": memory.get("exact_bytes_per_archived_frame"),
                    "compression_ratio": memory.get("compression_ratio"),
                    "layers": memory.get("storage_layers"),
                    "read_resident": next((t.get("read_resident") for t in reversed(trace) if t.get("read_resident")), None),
                    "chunks_replaced": sum(bool(t.get("attention_replaced")) for t in trace),
                    "last_retrieved": trace[-1].get("retrieved_frame_ids") if trace else None,
                    "read_decode_seconds": memory.get("read_decode_seconds_total"),
                    "elapsed_s": run.get("elapsed_s"),
                    "peak_gb": (run.get("peak_allocated_bytes") or 0) / 1e9,
                    "gpu": (run.get("hardware") or {}).get("gpu")})
                sheets.setdefault((job["task_id"], job["path"], job["seed"]), []).append((entry, ref, mid, end))
        rows.append(entry)
    (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    width, height = 256, 144
    for (task, path, seed), items in sheets.items():
        sheet = Image.new("RGB", (width * 3, (height + 16) * len(items)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (entry, ref, mid, end) in enumerate(items):
            y = i * (height + 16)
            ratio = entry.get("compression_ratio")
            draw.text((3, y + 2), f"{entry['variant']}  x{ratio:.1f}" if ratio else entry["variant"], fill="black")
            draw.text((width + 3, y + 2), "frame 12 | mid | final", fill="black")
            draw.text((2 * width + 3, y + 2), f"DINO vs f12 {entry['dino_end_vs_frame12']:.3f}", fill="black")
            for j, image in enumerate((ref, mid, end)):
                sheet.paste(image.resize((width, height)), (j * width, y + 16))
        sheet.save(args.output / f"{task}-{path}-seed{seed}.jpg", quality=85)
    measured = {(e["task"], e["path"]): e["bytes_per_frame"] for e in rows
                if e["variant"] == "exact" and e.get("bytes_per_frame")}
    for entry in rows:  # ratio against the exact job's measured bytes (token grid depends on the image)
        reference = measured.get((entry["task"], entry["path"])) or next(iter(measured.values()), None)
        if entry.get("bytes_per_frame") and reference:
            entry["compression_ratio"] = reference / entry["bytes_per_frame"]
    (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    by = {}
    for entry in rows:
        by.setdefault((entry["variant"], entry["seed"]), {})[(entry["task"], entry["path"])] = entry
    lines = ["| variant | MB per archived frame | ratio vs exact | GC007 hold | GC001 hold | GC007 turn_return | read | s (GC007 hold) | peak GB |",
             "|---|---:|---:|---:|---:|---:|---|---:|---:|"]

    def cell(entry):
        if not entry:
            return ""
        return f"{entry['dino_end_vs_frame12']:.3f}" if "dino_end_vs_frame12" in entry else entry["status"]

    for (variant, seed), cells in by.items():
        main_entry = cells.get(("GC007", "turn_hold_return")) or next(iter(cells.values()))
        size = main_entry.get("bytes_per_frame")
        ratio = main_entry.get("compression_ratio")
        lines.append(f"| {variant}{'' if seed == 42 else f' (seed {seed})'} | {size / 1e6:.0f} | {ratio:.1f}x |" if size else
                     f"| {variant}{'' if seed == 42 else f' (seed {seed})'} | 0 | n/a |")
        lines[-1] += (f" {cell(cells.get(('GC007', 'turn_hold_return')))} | {cell(cells.get(('GC001', 'turn_hold_return')))} |"
                      f" {cell(cells.get(('GC007', 'turn_return')))} | {main_entry.get('read_resident') or ''} |"
                      f" {main_entry.get('elapsed_s', 0) or 0:.0f} | {main_entry.get('peak_gb', 0) or 0:.1f} |")
    (args.output / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
