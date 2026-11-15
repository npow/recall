#!/usr/bin/env python3
"""Summarise a warm vs one-shot runner comparison: sanity of method effects, peak memory, speedup.

    python scripts/runner_summary.py ONESHOT.json WARM.json
"""
import json
import sys


def main(oneshot_path, warm_path):
    warm = json.load(open(warm_path))
    oneshot = json.load(open(oneshot_path))
    W = {r["id"]: r for r in warm["results"]}
    O = {r["id"]: r for r in oneshot["results"]}
    stock = next(r for key, r in W.items() if "-stock-" in key)["chunk_pixel_sha256"]
    for key, row in W.items():
        print(f"{key}: differs from stock = {row['chunk_pixel_sha256'] != stock}; peak {row['peak_allocated_bytes'] / 1e9:.1f} GB; "
              f"lora={((row['run'] or {}).get('lora') or {}).get('rank')}; warm_runtime={(row['run'] or {}).get('warm_runtime')}")
    for key, row in O.items():
        print(f"{key}: one-shot peak {row['peak_allocated_bytes'] / 1e9:.1f} GB")
    warm_seconds = [r["job_seconds"] for r in warm["results"]]
    oneshot_seconds = [r["job_seconds"] for r in oneshot["results"]]
    first = warm["results"][0]["client_wall_s"] - warm_seconds[0]
    mean_warm = sum(warm_seconds) / len(warm_seconds)
    mean_oneshot = sum(oneshot_seconds) / len(oneshot_seconds)
    print(f"warm startup (container start + load + warm-up) ~{first:.0f} s; mean warm job {mean_warm:.1f} s; mean one-shot job {mean_oneshot:.1f} s")
    print(f"jobs per slot-hour: one-shot {3600 / mean_oneshot:.1f} | warm steady-state {3600 / mean_warm:.1f} | "
          f"warm incl. one startup for {len(warm_seconds)} jobs {len(warm_seconds) * 3600 / (first + sum(warm_seconds)):.1f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
