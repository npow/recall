"""Paired analysis of the powered stock / budget-memory / 46-6 comparison (Gemini primary, Opus secondary).

Bootstrap CIs resample (case, seed) pairs. Aggregation follows the released PlayWorld rule per episode
(failed trajectory gate = 1.0). Only GC and out-of-sight OE tasks are present, so any overall figure is a
two-dimension mean, not the paper's four-dimension overall.
"""
import argparse
import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_gemini_scores import load, mean, fmt  # noqa: E402

ARMS = ["stock", "budget", "cache46"]


def bootstrap(values, n=10000, seed=0):
    if not values:
        return None
    rng = random.Random(seed)
    stats = sorted(mean([rng.choice(values) for _ in values]) for _ in range(n))
    return [stats[int(0.025 * n)], stats[int(0.975 * n)]]


def paired_table(rows, arm, key, base="stock", split=None):
    index = {(r["task"], r["seed"], r["arm"]): r for r in rows if r.get(key) is not None}
    pairs = [(index[(t, s, arm)], index[(t, s, base)]) for (t, s, a) in index
             if a == arm and (t, s, base) in index and (split is None or index[(t, s, a)]["split"] == split)]
    deltas = [a[key] - b[key] for a, b in pairs]
    return {"pairs": len(pairs), "mean_delta": mean(deltas), "ci95": bootstrap(deltas),
            "wins": sum(d > 0 for d in deltas), "ties": sum(d == 0 for d in deltas), "losses": sum(d < 0 for d in deltas)}


def interval(ci):
    return "n/a" if not ci else f"[{fmt(ci[0])}, {fmt(ci[1])}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--generated", type=Path, nargs="*", default=[], help="generated.json files (failure counts)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load(args.episodes, json.loads(args.meta.read_text()))
    scored = [r for r in rows if r.get("gemini") is not None]
    out = {"episodes_scored": len(scored), "judge_errors": [r["episode"] for r in rows if r.get("error")],
           "arms": {}, "paired": {}}
    for arm in ARMS:
        subset = [r for r in scored if r["arm"] == arm]
        if not subset:
            continue
        cats = {}
        for r in subset:
            for k, v in r["categories"].items():
                cats.setdefault(k, []).append(v)
        out["arms"][arm] = {
            "n": len(subset), "gemini_mean": mean([r["gemini"] for r in subset]),
            "gemini_ci95": bootstrap([r["gemini"] for r in subset]),
            "gc": mean([r["gemini"] for r in subset if r["split"] == "gc"]),
            "oe": mean([r["gemini"] for r in subset if r["split"] == "outsight"]),
            "gate_non_fail": sum(r["gemini_gate"] in ("pass", "partial") for r in subset),
            "opus_mean": mean([r["opus"] for r in subset]),
            "categories": {k: mean(v) for k, v in sorted(cats.items())}}
    for arm in ("budget", "cache46"):
        out["paired"][arm] = {"gemini_all": paired_table(scored, arm, "gemini"),
                              "gemini_gc": paired_table(scored, arm, "gemini", split="gc"),
                              "gemini_oe": paired_table(scored, arm, "gemini", split="outsight"),
                              "opus_all": paired_table(scored, arm, "opus")}
    failures = {}
    for path in args.generated:
        for g in json.loads(path.read_text()):
            arm = g.get("arm", "?")
            bad = bool(g.get("error") or g.get("returncode"))
            failures.setdefault(arm, {"generated": 0, "failed": 0})
            failures[arm]["generated"] += 1
            failures[arm]["failed"] += int(bad)
    out["generation_failures"] = failures
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "powered-analysis.json").write_text(json.dumps(out, indent=2))
    lines = ["| Arm | n | Gemini mean [95% CI] | GC | OE | gate non-fail | Opus mean |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for arm, s in out["arms"].items():
        lines.append(f"| {arm} | {s['n']} | {fmt(s['gemini_mean'])} {interval(s['gemini_ci95'])} | {fmt(s['gc'])} | "
                     f"{fmt(s['oe'])} | {s['gate_non_fail']}/{s['n']} | {fmt(s['opus_mean'])} |")
    lines += ["", "| Paired vs stock | slice | pairs | mean delta [95% CI] | W/T/L |", "| --- | --- | ---: | --- | --- |"]
    for arm, tables in out["paired"].items():
        for name, t in tables.items():
            lines.append(f"| {arm} | {name} | {t['pairs']} | {fmt(t['mean_delta'])} {interval(t['ci95'])} | "
                         f"{t['wins']}/{t['ties']}/{t['losses']} |")
    (args.output / "powered-table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({"generation_failures": failures, "judge_errors": out["judge_errors"]}, indent=2))


if __name__ == "__main__":
    main()
