"""Report for the full-run memory regression jobs (fixed paths through the warm runner).

Usage: python scripts/audit_regression_report.py RESULTS_JSON LOCAL_DIR OUT_JSON
LOCAL_DIR holds one directory per job id containing rollout.mp4 and run.json
(fetched from the Modal results volume). DINOv2-small cosine of the final frame
vs frame 12 (end of the initial hold) and vs frame 0; chunk-hash equality of the
repeated job (state leakage); eviction and retrieval trace summary.
"""
import json
import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def main():
    results, local, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    jobs = json.loads(results.read_text())
    jobs = jobs if isinstance(jobs, list) else jobs.get("results", jobs)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").eval()

    def embed(frame):
        with torch.no_grad():
            feature = model(**processor(images=Image.fromarray(frame), return_tensors="pt")).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(feature, dim=-1)[0]

    report = {}
    tiles = []
    for job in jobs:
        name = job["id"]
        directory = local / name
        run = json.loads((directory / "run.json").read_text())
        frames = np.stack(iio.mimread(directory / "rollout.mp4", memtest=False))
        first, ref, end = frames[0], frames[12], frames[-1]
        memory = run.get("memory") or {}
        trace = memory.get("trace", [])
        report[name] = {
            "error": job.get("error"), "gpu": job.get("gpu"), "job_seconds": job.get("job_seconds"),
            "peak_gb": (run.get("peak_allocated_bytes") or 0) / 1e9, "chunks": len(run["chunks"]),
            "dino_end_vs_frame12": float(embed(ref) @ embed(end)), "dino_end_vs_frame0": float(embed(first) @ embed(end)),
            "hashes": [chunk["pixel_sha256"] for chunk in run["chunks"]],
            "retrieval_calls": memory.get("retrieval_attention_calls"),
            "last_chunk_retrieved": trace[-1]["retrieved_frame_ids"] if trace else None,
            "last_chunk_archive_min_max": [min(trace[-1]["archive_frame_ids"]), max(trace[-1]["archive_frame_ids"])] if trace and trace[-1]["archive_frame_ids"] else None,
            "last_chunk_archive_size": len(trace[-1]["archive_frame_ids"]) if trace else None,
        }
        row = Image.new("RGB", (960, 184))
        for position, frame in enumerate((ref, frames[len(frames) // 2], end)):
            row.paste(Image.fromarray(frame).resize((320, 184)), (position * 320, 0))
        tiles.append((name, row))
    a, b = report.get("audit-budget-hold-a"), report.get("audit-budget-hold-b-repeat")
    leakage = None if not (a and b) else {"identical_chunk_hashes": a["hashes"] == b["hashes"],
                                          "matching_chunks": sum(x == y for x, y in zip(a["hashes"], b["hashes"])), "chunks": len(a["hashes"])}
    sheet = Image.new("RGB", (960, 184 * len(tiles)))
    for index, (_, row) in enumerate(tiles):
        sheet.paste(row, (0, 184 * index))
    sheet.save(out.with_suffix(".jpg"), quality=85)
    for value in report.values():
        value["hashes"] = value["hashes"][:2] + ["..."]
    out.write_text(json.dumps({"rows_in_image": [name for name, _ in tiles], "image_columns": "frame 12 | middle frame | final frame",
                               "jobs": report, "warm_repeat_leakage_check": leakage}, indent=1) + "\n")
    print(json.dumps({k: {m: (round(v[m], 3) if isinstance(v[m], float) else v[m]) for m in
        ("dino_end_vs_frame12", "dino_end_vs_frame0", "peak_gb", "retrieval_calls", "last_chunk_retrieved", "last_chunk_archive_min_max", "error")}
        for k, v in report.items()}, indent=1))
    print("leakage", leakage)


if __name__ == "__main__":
    main()
