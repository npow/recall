#!/usr/bin/env python3
"""Write a compact, committable summary of runner result files.

    python scripts/export_runner_validation.py OUT.json LABEL=RESULT.json [...]
"""
import json
import sys


def main(out, *specs):
    summary = {}
    for spec in specs:
        label, _, path = spec.partition("=")
        data = json.load(open(path))
        summary[label] = {"total_wall_s": data.get("total_wall_s"), "containers": data.get("containers"), "jobs": [
            {"id": r["id"], "gpu": ((r.get("run") or {}).get("hardware") or {}).get("gpu"), "status": r["status"],
             "job_seconds": round(r["job_seconds"], 1), "client_wall_s": round(r.get("client_wall_s", 0), 1),
             "peak_allocated_gb": round((r.get("peak_allocated_bytes") or 0) / 1e9, 1),
             "weights_unchanged": r.get("weights_unchanged"), "warm_runtime": (r.get("run") or {}).get("warm_runtime"),
             "argv": r["argv"], "result_directory": r["result_directory"], "chunk_pixel_sha256": r["chunk_pixel_sha256"]}
            for r in data["results"]]}
    with open(out, "w") as stream:
        json.dump(summary, stream, indent=1)
        stream.write("\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
