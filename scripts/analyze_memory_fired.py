"""Exploratory: did bounded pose memory actually act in the full run, and does the paired delta depend on it?

Reads a finished adaptive full run (episodes/*/run.json and gemini/episodes/*.json) and prints markdown tables.
Usage: python scripts/analyze_memory_fired.py <full-run dir, e.g. outputs/adaptive/full1>
"""
import collections
import glob
import json
import os
import random
import statistics as st
import sys

POWERED = "GC005 GC015 GC017 GC019 GC024 GC025 GC027 OE026 OE029 OE044 OE055 OE056 OE057 OE069".split()
LONG_PLAN = "GC005 OE044 GC015 OE056 OE055 GC025 GC019".split()
SPLITS = ("gc", "if", "insight", "outsight")


def load(root):
    def score(name):
        path = f"{root}/gemini/episodes/{name}.json"
        if not os.path.exists(path):
            return None
        data = json.load(open(path))
        official = (data.get("scores") or {}).get("official")
        return None if official is None else (float(official), (data.get("gate") or {}).get("verdict"))

    rows, ropes = [], collections.Counter()
    for run in sorted(glob.glob(f"{root}/episodes/budget-*/run.json")):
        name = os.path.basename(os.path.dirname(run))
        _, split, task, _ = name.split("-")
        data = json.load(open(run))
        memory = data.get("memory") or {}
        trace = memory.get("trace") or []
        ropes[memory.get("rope")] += 1
        stock_name = name.replace("budget-", "stock-")
        stock_run = f"{root}/episodes/{stock_name}/run.json"
        stock_seconds = json.load(open(stock_run)).get("generated_seconds") if os.path.exists(stock_run) else None
        budget, stock = score(name), score(stock_name)
        rows.append({"split": split, "task": task, "chunks": len(trace),
                     "fired": sum(1 for step in trace if step.get("attention_replaced")),
                     "sec": data.get("generated_seconds"), "ssec": stock_seconds,
                     "b": budget[0] if budget else None, "s": stock[0] if stock else None,
                     "bg": budget[1] if budget else None, "sg": stock[1] if stock else None,
                     "text": data.get("text_prompt_mode")})
    return rows, ropes


def ci(values, draws=10000):
    if len(values) < 2:
        return float("nan"), float("nan")
    means = sorted(st.mean(random.choices(values, k=len(values))) for _ in range(draws))
    return means[int(.025 * draws)], means[int(.975 * draws)]


def line(label, rows):
    deltas = [row["b"] - row["s"] for row in rows]
    if not deltas:
        return f"| {label} | 0 | n/a | n/a | n/a | n/a |"
    low, high = ci(deltas)
    wins, ties, losses = sum(d > 0 for d in deltas), sum(d == 0 for d in deltas), sum(d < 0 for d in deltas)
    return (f"| {label} | {len(deltas)} | {st.mean(row['s'] for row in rows):.2f} | {st.mean(row['b'] for row in rows):.2f} "
            f"| {st.mean(deltas):+.2f} [{low:+.2f}, {high:+.2f}] | {wins}/{ties}/{losses} |")


def main(root):
    random.seed(0)
    rows, ropes = load(root)
    out = [f"Budget episodes: {len(rows)}. RoPE setting recorded in run.json: {dict(ropes)}.", "",
           "| dimension | budget eps | eps with any retrieval | median chunks fired / total | median length s, budget | median length s, stock | budget eps <= 4.5 s |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for split in SPLITS:
        sub = [row for row in rows if row["split"] == split]
        out.append(f"| {split} | {len(sub)} | {sum(row['fired'] > 0 for row in sub)} "
                   f"| {st.median(row['fired'] for row in sub):.0f} / {st.median(row['chunks'] for row in sub):.0f} "
                   f"| {st.median(row['sec'] for row in sub):.1f} | {st.median(row['ssec'] for row in sub if row['ssec']):.1f} "
                   f"| {sum(row['sec'] <= 4.5 for row in sub)} |")
    paired = [row for row in rows if row["b"] is not None and row["s"] is not None]
    out += ["", f"Paired cases (both arms scored): {len(paired)} of {len(rows)}. Unscored in either arm: "
            f"{[(row['split'], row['task']) for row in rows if row['b'] is None or row['s'] is None]}.", "",
            "| slice | pairs | stock mean | budget mean | paired delta [95% CI] | W/T/L |", "|---|---:|---:|---:|---|---|",
            line("all", paired),
            line("retrieval fired (>= 1 chunk)", [row for row in paired if row["fired"] > 0]),
            line("retrieval never fired", [row for row in paired if row["fired"] == 0]),
            line("fired in >= 5 chunks", [row for row in paired if row["fired"] >= 5])]
    for split in SPLITS:
        out.append(line(f"{split}: all", [row for row in paired if row["split"] == split]))
        out.append(line(f"{split}: fired", [row for row in paired if row["split"] == split and row["fired"] > 0]))
        out.append(line(f"{split}: not fired", [row for row in paired if row["split"] == split and row["fired"] == 0]))
    cohort = [row for row in paired if row["task"] in POWERED and row["split"] in ("gc", "outsight")]
    out += [line("earlier powered cohort (14 cases), in this run", cohort),
            line("powered cohort: GC", [row for row in cohort if row["split"] == "gc"]),
            line("powered cohort: out-of-sight", [row for row in cohort if row["split"] == "outsight"]),
            line("powered cohort: 7 long-plan cases", [row for row in cohort if row["task"] in LONG_PLAN]),
            line("powered cohort: 7 original cases", [row for row in cohort if row["task"] not in LONG_PLAN]),
            "", "| case | stock | budget | chunks fired / total | budget s | stock s | stock gate | budget gate |",
            "|---|---:|---:|---:|---:|---:|---|---|"]
    for row in sorted(cohort, key=lambda row: row["task"]):
        out.append(f"| {row['task']} | {row['s']} | {row['b']} | {row['fired']}/{row['chunks']} | {row['sec']:.1f} "
                   f"| {row['ssec']:.1f} | {row['sg']} | {row['bg']} |")
    out += ["", f"Gate verdicts, paired cases: stock {dict(collections.Counter(row['sg'] for row in paired))}; "
            f"budget {dict(collections.Counter(row['bg'] for row in paired))}.",
            f"Score exactly 1.0: stock {sum(row['s'] == 1.0 for row in paired)}, budget {sum(row['b'] == 1.0 for row in paired)} of {len(paired)}.",
            f"Mean episode length: stock {st.mean(row['ssec'] for row in paired):.1f} s, budget {st.mean(row['sec'] for row in paired):.1f} s; "
            f"pairs with identical length: {sum(abs(row['sec'] - row['ssec']) < 1e-6 for row in paired)} of {len(paired)}."]
    out += ["", *placebo(root)]
    print("\n".join(out))


def placebo(root):
    """Pairs where retrieval never fired and both arms produced identical pixels: any score gap is judge noise."""
    identical = []
    for run in sorted(glob.glob(f"{root}/episodes/budget-*/run.json")):
        name = os.path.basename(os.path.dirname(run))
        stock_name = name.replace("budget-", "stock-")
        try:
            budget, stock = json.load(open(run)), json.load(open(f"{root}/episodes/{stock_name}/run.json"))
            budget_score = json.load(open(f"{root}/gemini/episodes/{name}.json"))["scores"]["official"]
            stock_score = json.load(open(f"{root}/gemini/episodes/{stock_name}.json"))["scores"]["official"]
        except (OSError, KeyError, TypeError):
            continue
        if budget_score is None or stock_score is None:
            continue
        hashes = lambda data: [chunk.get("pixel_sha256") for chunk in data["chunks"]]
        if hashes(budget) == hashes(stock):
            identical.append((name.split("-", 1)[1], stock_score, budget_score))
    gaps = [abs(b - s) for _, s, b in identical]
    lines = [f"Identical-pixel pairs (every chunk's pixel hash equal in both arms): {len(identical)}. "
             f"Pairs scored differently by the judge: {sum(g > 0 for g in gaps)}; mean absolute gap "
             f"{st.mean(gaps):.2f}, largest {max(gaps):.2f}; mean signed budget-minus-stock "
             f"{st.mean(b - s for _, s, b in identical):+.2f}.", "",
             "| identical-pixel pair scored differently | stock | budget |", "|---|---:|---:|"]
    lines += [f"| {name} | {s} | {b} |" for name, s, b in identical if s != b]
    return lines


if __name__ == "__main__":
    main(sys.argv[1])
