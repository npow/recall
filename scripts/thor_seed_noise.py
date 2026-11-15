"""Seed-to-seed noise of the stock arm: Thor seed 43 vs the full run's seed 42, same cases, same Gemini judge.

Usage: python scripts/thor_seed_noise.py [--seed42-dir DIR] [--write reproduction/seed-noise-thor.md]
Reads outputs/thor-seed43/gemini/episodes/stock-*.json and the full run's
outputs/adaptive/full1/gemini/episodes/stock-*-seed42.json. Only cases scored in both seeds are compared.
"""
import argparse
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED42 = Path("/home/npow/code/iteratehack/.claude/worktrees/agent-a20dd1a4d88b1a928/outputs/adaptive/full1/gemini/episodes")
THOR = ROOT / "outputs/thor-seed43/gemini/episodes"


def load(directory, suffix):
    rows = {}
    for path in Path(directory).glob(f"stock-*-{suffix}.json"):
        data = json.loads(path.read_text())
        if data.get("error"):
            continue
        score = (data.get("scores") or {}).get("official")
        if score is None:
            continue
        _, split, task, _ = path.stem.split("-")
        rows[(split, task)] = {"score": float(score), "gate": (data.get("gate") or {}).get("verdict")}
    return rows


def summarize(seed42, seed43):
    common = sorted(set(seed42) & set(seed43))
    pairs = [(key, seed42[key], seed43[key]) for key in common]
    diffs = [b["score"] - a["score"] for _, a, b in pairs]
    out = {"pairs": len(pairs), "cases": [], "by_split": {}}
    for (split, task), a, b in pairs:
        out["cases"].append({"split": split, "task": task, "seed42": a["score"], "seed43": b["score"],
                             "diff": b["score"] - a["score"], "gate42": a["gate"], "gate43": b["gate"]})
    if len(diffs) >= 2:
        out["mean_diff"] = statistics.fmean(diffs)
        out["sd_of_diff"] = statistics.stdev(diffs)
        out["per_episode_sd"] = out["sd_of_diff"] / math.sqrt(2)  # SD of one episode's score if both seeds are exchangeable
        out["mean_abs_diff"] = statistics.fmean(abs(d) for d in diffs)
        out["gate_flips"] = sum(1 for _, a, b in pairs if a["gate"] != b["gate"])
        out["floor_both"] = sum(1 for _, a, b in pairs if a["score"] == 1.0 and b["score"] == 1.0)
    for split in sorted({key[0] for key, _, _ in pairs}):
        sub = [b["score"] - a["score"] for (s, _), a, b in pairs if s == split]
        out["by_split"][split] = {"n": len(sub), "mean_diff": statistics.fmean(sub),
                                  "mean_abs_diff": statistics.fmean(abs(d) for d in sub)}
    return out


def render(summary):
    lines = ["# Seed noise of the stock arm (Thor seed 43 vs full-run seed 42)", "",
             "Same cases, same frozen flags, same Gemini Vertex judge; hardware differs (Thor vs Modal H100/H200), so some",
             "of the difference is GPU numerics, not only the seed. Player trajectories differ between seeds.", "",
             f"Pairs scored in both seeds: **{summary['pairs']}**", ""]
    if "sd_of_diff" in summary:
        lines += [f"- Mean difference (seed43 - seed42): {summary['mean_diff']:+.2f}",
                  f"- SD of per-case difference: {summary['sd_of_diff']:.2f} (per-episode SD about {summary['per_episode_sd']:.2f})",
                  f"- Mean absolute difference: {summary['mean_abs_diff']:.2f}",
                  f"- Gate verdict flips between seeds: {summary['gate_flips']} of {summary['pairs']}; both at the 1.0 floor: {summary['floor_both']}", ""]
    lines += ["| split | case | seed 42 | seed 43 | diff | gate 42 | gate 43 |", "|---|---|---:|---:|---:|---|---|"]
    for row in summary["cases"]:
        lines.append(f"| {row['split']} | {row['task']} | {row['seed42']:.2f} | {row['seed43']:.2f} | {row['diff']:+.2f} | {row['gate42']} | {row['gate43']} |")
    lines += ["", "| split | n | mean diff | mean abs diff |", "|---|---:|---:|---:|"]
    for split, row in summary["by_split"].items():
        lines.append(f"| {split} | {row['n']} | {row['mean_diff']:+.2f} | {row['mean_abs_diff']:.2f} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed42-dir", type=Path, default=DEFAULT_SEED42)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    summary = summarize(load(args.seed42_dir, "seed42"), load(THOR, "seed43"))
    text = render(summary)
    print(text)
    if args.write:
        args.write.write_text(text)
        (args.write.with_suffix(".json")).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
