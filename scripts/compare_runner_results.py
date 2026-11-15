#!/usr/bin/env python3
"""Bit-exactness and timing comparison between runner result files.

    python scripts/compare_runner_results.py REFERENCE.json:oneshot WARM.json:warm [...]

Jobs are paired by their `pair` key: the id with its leading `<prefix>-` removed
(`w1-stock-GC007-s42` pairs with `o1-stock-GC007-s42` and with `w4-stock-GC007-s42-repeat`
if the id ends in `-repeat`). Prints per-pair chunk-hash equality and a timing table.
"""
import json
import re
import sys
from pathlib import Path


def pair_key(job_id):
    key = re.sub(r"^[a-z]+\d+-", "", job_id or "")
    return re.sub(r"-repeat\d*$", "", key)


def load(spec):
    path, _, label = spec.partition(":")
    data = json.loads(Path(path).read_text())
    return label or Path(path).stem, data


def main(specs):
    groups = [load(spec) for spec in specs]
    table = {}
    for label, data in groups:
        for result in data["results"]:
            table.setdefault(pair_key(result["id"]), []).append((label, result))
    ok = True
    for key, entries in sorted(table.items()):
        hashes = {tuple(result["chunk_pixel_sha256"]) for _, result in entries}
        identical = len(hashes) == 1 and all(r["chunk_pixel_sha256"] for _, r in entries)
        ok &= identical
        gpus = {((r.get("run") or {}).get("hardware") or {}).get("gpu") for _, r in entries}
        detail = ", ".join(f"{r['id']}[{label}] {len(r['chunk_pixel_sha256'])} chunks {r['job_seconds']:.0f}s "
                           f"on {((r.get('run') or {}).get('hardware') or {}).get('gpu')}" for label, r in entries)
        if len(gpus) > 1:
            detail += "  (DIFFERENT GPU MODELS: not a valid bit-exactness comparison)"
        print(f"{'IDENTICAL' if identical else 'DIFFERENT'}  {key}: {detail}")
    for label, data in groups:
        seconds = [r["job_seconds"] for r in data["results"]]
        print(f"{label}: jobs={len(seconds)} total_wall_s={data.get('total_wall_s', 0):.0f} "
              f"job_seconds={[round(s) for s in seconds]} weights_unchanged="
              f"{[r.get('weights_unchanged') for r in data['results']]}")
    print("ALL BIT-IDENTICAL" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
