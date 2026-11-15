"""Final analysis of the full 171-case run (label full1): paper formula, CIs, paired deltas, GC013 sensitivity.

    python scripts/analyze_full_run.py --label full1 --out reproduction/full-run

Reads outputs/adaptive/<label>/gemini/episodes/<arm>-<split>-<TASK>-seed<N>.json (Gemini, released rubric
via Vertex) and, for generation failures, outputs/adaptive/<label>/status.json. No API calls.
overall = equal mean of the four dimension means (GC, IF, insight, out-of-sight). Bootstrap CIs resample
cases within each dimension. A dimension with no scored cases makes overall unavailable (never partial).
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worldsync import fourdim  # noqa: E402

ARMS = ("stock", "budget")
EXPECTED = {"gc": 48, "if": 50, "insight": 30, "outsight": 43}
EXCLUDED_SENSITIVITY = {("gc", "GC013")}
# Insight development cases used for the pre-declared text-prompt amendment (reproduction/full-run-config.json):
# the held-out insight set is the other 24 of 30.
INSIGHT_DEV = {("insight", task) for task in ("OE008", "OE022", "OE035", "OE040", "OE063", "OE074")}


def load(label):
    base = ROOT / "outputs/adaptive" / label
    rows = []
    for path in sorted((base / "gemini/episodes").glob("*.json")):
        arm, split, task = path.stem.split("-")[:3]
        result = json.loads(path.read_text())
        official = (result.get("scores") or {}).get("official")
        rows.append({"arm": arm, "split": split, "task": task, "score": official,
                     "error": result.get("error"), "gate": (result.get("gate") or {}).get("verdict"),
                     "revisit_required": (result.get("gate") or {}).get("revisit_required"),
                     "revisit_observed": (result.get("gate") or {}).get("revisit_observed")})
    status = json.loads((base / "status.json").read_text()) if (base / "status.json").is_file() else {"arms": {}}
    return rows, status


def case_scores(rows, arm, drop=()):
    return {(r["split"], r["task"]): float(r["score"]) for r in rows
            if r["arm"] == arm and r["score"] is not None and not r["error"] and (r["split"], r["task"]) not in drop}


def summarize(rows, drop=()):
    out = {"arms": {}, "paired": None}
    scores = {arm: case_scores(rows, arm, drop) for arm in ARMS}
    for arm in ARMS:
        dims = fourdim.dimension_means(scores[arm])
        out["arms"][arm] = {"dimension_means": dims, "overall": fourdim.overall(dims),
                            "overall_ci95": fourdim.bootstrap_overall(scores[arm]),
                            "cases_scored": {s: sum(1 for (sp, _) in scores[arm] if sp == s) for s in fourdim.SPLITS}}
    out["paired"] = fourdim.bootstrap_paired_delta(scores["budget"], scores["stock"])
    return out


def gates(rows):
    table = {}
    for arm in ARMS:
        for split in fourdim.SPLITS:
            sub = [r for r in rows if r["arm"] == arm and r["split"] == split and r["score"] is not None and not r["error"]]
            table[f"{arm}/{split}"] = {"n": len(sub), **{v: sum(r["gate"] == v for r in sub) for v in ("fail", "partial", "pass")},
                                       "score_exactly_1.0": sum(r["score"] == 1.0 for r in sub),
                                       "revisit_required": sum(r["revisit_required"] == "yes" for r in sub),
                                       "revisit_observed": sum(r["revisit_observed"] == "yes" for r in sub)}
    return table


def fmt(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def interval(ci):
    return "n/a" if not ci else f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def render(full, without, table, status, rows, held_out=None, sensitivity=()):
    paper = fourdim.PAPER
    head = "| | " + " | ".join(fourdim.DIMENSION_NAMES[s] for s in fourdim.SPLITS) + " | overall (equal mean of four) |"
    lines = [head, "|---|" + "---:|" * 5,
             "| paper, LingBot-World2 (hosted) | " + " | ".join(fmt(paper[s]) for s in fourdim.SPLITS) + f" | {fmt(paper['overall'])} |",
             f"| paper, Genie 3 (closed) | | | | | {fmt(paper['genie3_overall'])} |"]
    variants = [("all 171 cases", full), ("without GC013", without)]
    if held_out is not None:
        variants.append(("insight = 24 held-out cases (6 dev cases dropped)", held_out))
    for label, data in variants:
        for arm in ARMS:
            d = data["arms"][arm]
            n = "/".join(str(d["cases_scored"][s]) for s in fourdim.SPLITS)
            lines.append(f"| ours {arm}, {label} (scored {n}) | " + " | ".join(fmt(d["dimension_means"][s]) for s in fourdim.SPLITS)
                         + f" | {fmt(d['overall'])} {interval(d['overall_ci95'])} |")
    lines += ["", "Paired budget - stock (cases scored in both arms; overall = equal mean of the four dimension deltas):", ""]
    for label, data in [("all 171", full), ("without GC013", without)] + ([("insight 24 held-out", held_out)] if held_out else []):
        p = data["paired"]
        if not p:
            lines.append(f"* {label}: not available (a dimension has no paired cases)")
            continue
        per = ", ".join(f"{fourdim.DIMENSION_NAMES[s]} {p['per_dimension'][s]:+.2f} (n={p['pairs'][s]})" for s in fourdim.SPLITS)
        lines.append(f"* {label}: overall {p['delta']:+.2f} {interval(p['ci95'])}; {per}")
    lines += ["", "Gemini gate outcomes and the score-1.0 floor, per arm and dimension:", "",
              "| arm/dimension | scored | gate fail | partial | pass | score exactly 1.0 | revisit required | revisit observed |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, g in table.items():
        lines.append(f"| {key} | {g['n']} | {g['fail']} | {g['partial']} | {g['pass']} | {g['score_exactly_1.0']} | "
                     f"{g['revisit_required']} | {g['revisit_observed']} |")
    failures = {arm: len(status["arms"].get(arm, {}).get("failed", [])) for arm in ARMS}
    unscored = {arm: sum(1 for r in rows if r["arm"] == arm and (r["score"] is None or r["error"])) for arm in ARMS}
    if sensitivity:
        lines += ["", "Kept separately as a 'both'-prompt sensitivity datum (not in any table above): " + ", ".join(sensitivity)]
    lines += ["", f"Generation failures after retries: {failures}. Episodes generated but not scored (judge errors): {unscored}.",
              "Expected episodes per arm and dimension: " + json.dumps(EXPECTED) + ".",
              "", "A claim of beating 2.12 requires the lower CI bound of the overall above 2.12 (reproduction/pass-criterion.md)."]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="full1")
    parser.add_argument("--out", type=Path, default=ROOT / "reproduction/full-run")
    args = parser.parse_args()
    rows, status = load(args.label)
    full = summarize(rows)
    without = summarize(rows, EXCLUDED_SENSITIVITY)
    held_out = summarize(rows, INSIGHT_DEV)  # insight on the 24 held-out cases (dev cases dropped)
    sens_dir = ROOT / "outputs/adaptive" / args.label / "sensitivity_both/gemini/episodes"
    sensitivity = sorted(p.stem for p in sens_dir.glob("*.json")) if sens_dir.is_dir() else []
    table = gates(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "analysis.json").write_text(json.dumps({"all": full, "without_gc013": without, "insight_24_held_out": held_out,
                                                       "both_prompt_sensitivity_episodes": sensitivity, "gates": table,
                                                       "paper": fourdim.PAPER}, indent=2) + "\n")
    (args.out / "per-episode.csv").write_text("arm,split,task,gate,score,revisit_required,revisit_observed,error\n" + "\n".join(
        f"{r['arm']},{r['split']},{r['task']},{r['gate']},{r['score']},{r['revisit_required']},{r['revisit_observed']},{bool(r['error'])}"
        for r in sorted(rows, key=lambda r: (r['arm'], r['split'], r['task']))) + "\n")
    text = render(full, without, table, status, rows, held_out, sensitivity)
    (args.out / "table.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
