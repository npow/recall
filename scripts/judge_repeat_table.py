"""Render reproduction/judge-repeat/README.md from summary.json and per-episode.csv (no API calls)."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reproduction/judge-repeat"
NAMES = {"gc": "GC", "if": "IF", "insight": "insight", "outsight": "out-of-sight"}


def main():
    d = json.loads((OUT / "summary.json").read_text())
    rows = list(csv.DictReader(open(OUT / "per-episode.csv")))
    lines = ["# Judge-repeat sensitivity analysis (Gemini, 3 calls per episode)", "",
             "Sensitivity analysis only. The frozen single-call results in `reproduction/full-run/` stay the primary numbers.",
             "Out-of-sight and GC episodes of the full run (both arms) that scored exactly 1.0 or had a gate verdict of "
             "fail / partial were re-judged with 2 extra independent Gemini calls (same released prompts, rubric, gate, "
             "sampling and model; separate caches). **Selection is one-sided**: floor episodes can only go up, and "
             "gate-affected ones were chosen because their first call was low, so the repeated means are upper-biased "
             "(regression to the mean). This measures whether the judge is *consistent*, not whether it is *right*.", "",
             f"Episodes with both repeats: {d['episodes_with_two_repeats']}.", "",
             "| arm | dimension | n | single-call mean | mean of 3 | mean per-episode SD | majority-gate flips | lower | higher |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ("stock", "budget"):
        for split in ("gc", "outsight"):
            sub = [r for r in rows if r["arm"] == arm and r["split"] == split]
            if not sub:
                continue
            single = sum(float(r["single_score"]) for r in sub) / len(sub)
            mean3 = sum(float(r["mean3"]) for r in sub) / len(sub)
            sd = sum(float(r["sd3"]) for r in sub) / len(sub)
            lower = sum(float(r["mean3"]) < float(r["single_score"]) - 1e-9 for r in sub)
            higher = sum(float(r["mean3"]) > float(r["single_score"]) + 1e-9 for r in sub)
            flips = sum(r["gate_flip"] == "True" for r in sub)
            lines.append(f"| {arm} | {NAMES[split]} | {len(sub)} | {single:.2f} | {mean3:.2f} | {sd:.2f} | {flips} | {lower} | {higher} |")
    floor = [r for r in rows if r["split"] == "outsight" and float(r["single_score"]) == 1.0]
    stay = sum(float(r["rep1"]) == 1.0 and float(r["rep2"]) == 1.0 for r in floor)
    lines += ["", f"Out-of-sight floor: {stay} of {len(floor)} episodes that scored exactly 1.0 score 1.0 on all three calls.", "",
              "Effect on the full-run table when the repeated episodes use their mean of 3 (others keep their single call):", "",
              "| | GC | IF | insight | out-of-sight | overall |", "|---|---:|---:|---:|---:|---:|"]
    for key, label in (("single_call_primary", "single call (primary)"),
                       ("repeated_mean_of_3_replacing_single_call_on_repeated_episodes", "with repeats")):
        for arm in ("stock", "budget"):
            x = d[key][arm]
            lines.append(f"| {arm}, {label} | " + " | ".join(f"{x['dims'][s]:.2f}" for s in NAMES) + f" | {x['overall']:.2f} |")
    lines += ["", "Paired budget - stock (overall = equal mean of the four dimension deltas, bootstrap over cases):", ""]
    for key, label in (("paired_budget_minus_stock_single_call", "single call"), ("paired_budget_minus_stock_with_repeats", "with repeats")):
        p = d[key]
        per = ", ".join(f"{NAMES[s]} {p['per_dimension'][s]:+.2f}" for s in NAMES)
        lines.append(f"* {label}: {p['delta']:+.2f} [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]; {per}")
    lines += ["", "Per-episode values: [per-episode.csv](per-episode.csv). Machine-readable summary: [summary.json](summary.json)."]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
