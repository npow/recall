"""Four-dimension analysis: per-dimension means, equal-mean overall, paired delta, with the paper's numbers alongside.

  python scripts/analyze_fourdim.py OUT_DIR SEED_DIR [SEED_DIR ...]

Gemini (released rubric via Vertex) is the primary score; Opus (worldsync.judge) is secondary. Episodes
without a score are excluded and counted. Seeds are averaged per (case, arm) before aggregating, and
bootstrap CIs resample cases within each dimension.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_gemini_scores import load  # noqa: E402
from worldsync import fourdim  # noqa: E402

ARMS = ["stock", "budget"]


def case_scores(rows, arm, key):
    """{(split, task): mean over seeds of rows[key]} for one arm."""
    bucket = {}
    for r in rows:
        if r.get("arm") == arm and r.get(key) is not None:
            bucket.setdefault((r["split"], r["task"]), []).append(r[key])
    return {case: sum(v) / len(v) for case, v in bucket.items()}


def fmt(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def interval(ci):
    return "n/a" if not ci else f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def analyse(seed_dirs):
    rows, generated = [], []
    for seed_dir in seed_dirs:
        meta = json.loads((Path(seed_dir) / "gemini/meta.json").read_text())
        rows += load(Path(seed_dir) / "gemini/episodes", meta)
        generated += json.loads((Path(seed_dir) / "generated.json").read_text())
    scored = [r for r in rows if r.get("gemini") is not None]
    out = {"episodes_scored": len(scored), "errors": [r["episode"] for r in rows if r.get("error")],
           "generation_failures": [{k: g.get(k) for k in ("arm", "split", "task", "seed", "returncode", "error")}
                                   for g in generated if g.get("returncode") or g.get("error")],
           "paper": fourdim.PAPER, "arms": {}, "paired": {}}
    for judge in ("gemini", "opus"):
        for arm in ARMS:
            scores = case_scores(scored if judge == "gemini" else [r for r in rows if r.get("opus") is not None], arm, judge)
            dims = fourdim.dimension_means(scores)
            out["arms"].setdefault(arm, {})[judge] = {
                "dimension_means": dims, "overall": fourdim.overall(dims), "overall_ci95": fourdim.bootstrap_overall(scores),
                "cases_scored": {s: sum(1 for (sp, _) in scores if sp == s) for s in fourdim.SPLITS}}
        a, b = case_scores(rows, "budget", judge), case_scores(rows, "stock", judge)
        out["paired"][judge] = fourdim.bootstrap_paired_delta(a, b)
    gates = {}
    for arm in ARMS:
        subset = [r for r in scored if r["arm"] == arm]
        for split in fourdim.SPLITS:
            sub = [r for r in subset if r["split"] == split]
            gates[f"{arm}/{split}"] = {"n": len(sub), "gate_fail": sum(r["gemini_gate"] == "fail" for r in sub),
                                       "gate_partial": sum(r["gemini_gate"] == "partial" for r in sub),
                                       "gate_pass": sum(r["gemini_gate"] == "pass" for r in sub)}
    out["gates"] = gates
    return out


def render(out):
    paper = out["paper"]
    lines = ["| | " + " | ".join(fourdim.DIMENSION_NAMES[s] for s in fourdim.SPLITS) + " | overall (equal mean of four) |",
             "|---|" + "---:|" * 5]
    lines.append("| paper, LingBot-World2 (hosted) | " + " | ".join(fmt(paper[s]) for s in fourdim.SPLITS)
                 + f" | {fmt(paper['overall'])} |")
    lines.append(f"| paper, Genie 3 (closed) | | | | | {fmt(paper['genie3_overall'])} |")
    for judge, label in (("gemini", "Gemini (primary)"), ("opus", "Opus (secondary)")):
        for arm in ARMS:
            data = out["arms"][arm][judge]
            n = "/".join(str(data["cases_scored"][s]) for s in fourdim.SPLITS)
            lines.append(f"| ours {arm}, {label} (cases {n}) | "
                         + " | ".join(fmt(data["dimension_means"][s]) for s in fourdim.SPLITS)
                         + f" | {fmt(data['overall'])} {interval(data['overall_ci95'])} |")
    lines += ["", "Paired budget - stock (cases scored in both arms, overall = equal mean of four dimension deltas):", ""]
    for judge in ("gemini", "opus"):
        p = out["paired"][judge]
        if not p:
            lines.append(f"* {judge}: not yet available (a dimension has no paired cases)")
            continue
        per = ", ".join(f"{fourdim.DIMENSION_NAMES[s]} {p['per_dimension'][s]:+.2f} (n={p['pairs'][s]})" for s in fourdim.SPLITS)
        lines.append(f"* {judge}: overall {p['delta']:+.2f} {interval(p['ci95'])}; {per}")
    lines += ["", "Gemini gate outcomes (fail/partial/pass of n):", ""]
    for key, g in out["gates"].items():
        lines.append(f"* {key}: {g['gate_fail']}/{g['gate_partial']}/{g['gate_pass']} of {g['n']}")
    lines += ["", f"Scored episodes: {out['episodes_scored']}; judge errors: {len(out['errors'])}; "
              f"generation failures: {len(out['generation_failures'])}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("seed_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    out = analyse(args.seed_dirs)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "fourdim-analysis.json").write_text(json.dumps(out, indent=2))
    text = render(out)
    (args.output / "fourdim-table.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
