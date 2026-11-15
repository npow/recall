"""Aggregate Gemini-rescored episodes per arm and compare with our Claude (Opus) judge.

Aggregation follows the released PlayWorld rule: a failed trajectory gate scores 1.0,
otherwise the rubric score (GC: category-weighted with identity_id weight 2).
Only GC and out-of-sight OE cases are present, so the overall figure is a two-dimension
mean, not the paper's four-dimension overall.
"""
import argparse
import json
import math
from pathlib import Path

ARMS = ["stock", "cache46", "fb", "fbcomp", "v3stock"]


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            result[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return result


def pearson(x, y):
    if len(x) < 3:
        return None
    mx, my = mean(x), mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else None


def fmt(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def load(episodes_dir, meta):
    rows = []
    for path in sorted(Path(episodes_dir).glob("*.json")):
        result = json.loads(path.read_text())
        if result.get("error") or "scores" not in result:
            rows.append({"episode": path.stem, "error": result.get("error", "no scores")})
            continue
        info = meta.get(path.stem)
        if not info:
            continue
        gate = result.get("gate") or {}
        rubric = result.get("rubric") or {}
        categories = rubric.get("categories") or rubric.get("category_scores_1_to_5") or {}
        rows.append({
            "episode": path.stem, "arm": info["arm"], "seed": info["seed"], "task": info["task"], "split": info["split"],
            "gemini_gate": gate.get("verdict"), "gemini": result["scores"]["official"],
            "gemini_final": result["scores"]["final_score"],
            "categories": {k: v.get("score") for k, v in categories.items() if isinstance(v, dict)},
            "opus_gate": info["opus"].get("verdict"), "opus": info["opus"].get("score"),
            "tokens": sum((c.get("prompt_tokens") or 0) + (c.get("output_tokens") or 0) for c in result.get("calls", [])),
        })
    return rows


def arm_summary(rows):
    summary = {}
    for arm in ARMS:
        subset = [r for r in rows if r.get("arm") == arm and r.get("gemini") is not None]
        if not subset:
            continue
        gc = [r["gemini"] for r in subset if r["split"] == "gc"]
        oe = [r["gemini"] for r in subset if r["split"] == "outsight"]
        gate_pass = [r for r in subset if r["gemini_gate"] in ("pass", "partial")]
        categories = {}
        for r in subset:
            for name, value in r["categories"].items():
                categories.setdefault(name, []).append(value)
        summary[arm] = {
            "n": len(subset), "gc": mean(gc), "n_gc": len(gc), "oe_out_of_sight": mean(oe), "n_oe": len(oe),
            "two_dim_mean": mean([mean(gc), mean(oe)]), "gate_non_fail": len(gate_pass),
            "opus_mean": mean([r["opus"] for r in subset]),
            "category_means": {k: mean(v) for k, v in sorted(categories.items())},
        }
    return summary


def paired(rows, arm, base="stock"):
    index = {(r["task"], r["seed"], r["arm"]): r for r in rows if r.get("gemini") is not None}
    pairs = [(index[(t, s, arm)], index[(t, s, base)]) for (t, s, a) in index if a == arm and (t, s, base) in index]
    deltas = [a["gemini"] - b["gemini"] for a, b in pairs]
    return {"pairs": len(pairs), "mean_delta": mean(deltas), "wins": sum(d > 0 for d in deltas),
            "ties": sum(d == 0 for d in deltas), "losses": sum(d < 0 for d in deltas),
            "opus_mean_delta": mean([a["opus"] - b["opus"] for a, b in pairs if a["opus"] is not None and b["opus"] is not None])}


def agreement(rows):
    both = [r for r in rows if r.get("gemini") is not None and r.get("opus") is not None]
    g, o = [r["gemini"] for r in both], [r["opus"] for r in both]
    confusion = {}
    for r in both:
        key = f"opus={r['opus_gate']} gemini={r['gemini_gate']}"
        confusion[key] = confusion.get(key, 0) + 1
    return {"n": len(both), "pearson": pearson(g, o), "spearman": pearson(ranks(g), ranks(o)),
            "bias_gemini_minus_opus": mean([a - b for a, b in zip(g, o)]),
            "gate_pairs": dict(sorted(confusion.items())),
            "gate_pass_rate": {"gemini": mean([r["gemini_gate"] in ("pass", "partial") for r in both]),
                               "opus": mean([r["opus_gate"] in ("pass", "partial") for r in both])}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load(args.episodes, json.loads(args.meta.read_text()))
    scored = [r for r in rows if r.get("gemini") is not None]
    report = {"episodes_scored": len(scored), "errors": [r for r in rows if r.get("error")],
              "arms": arm_summary(rows), "paired_vs_stock": {arm: paired(rows, arm) for arm in ("cache46", "fb", "fbcomp")},
              "agreement": agreement(rows), "total_tokens": sum(r["tokens"] for r in scored)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "gemini-results.json").write_text(json.dumps({"report": report, "episodes": scored}, indent=2))
    lines = ["| Arm | n | GC (n) | Out-of-sight OE (n) | 2-dim mean | Gate non-fail | Opus mean (same eps) |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, s in report["arms"].items():
        lines.append(f"| {arm} | {s['n']} | {fmt(s['gc'])} ({s['n_gc']}) | {fmt(s['oe_out_of_sight'])} ({s['n_oe']}) | {fmt(s['two_dim_mean'])} | {s['gate_non_fail']}/{s['n']} | {fmt(s['opus_mean'])} |")
    (args.output / "arm-table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({k: report[k] for k in ("paired_vs_stock", "agreement", "total_tokens", "episodes_scored")}, indent=2))


if __name__ == "__main__":
    main()
