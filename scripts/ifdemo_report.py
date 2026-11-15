"""Paired IF report: baseline (frozen legacy player) vs canonical controller over all judged IF tasks.

    python scripts/ifdemo_report.py [--output reproduction/if-paired-comparison.md]

Per video: mean of the judged Gemini calls (up to 3). A task counts only when both arms have 3 scored calls.
Pre-declared criterion (written before the scores existed): paired mean difference with a task-bootstrap 95% interval,
plus instruction-gate pass/fail counts. Demo clips: tasks where every corrected call beats every baseline call.
"""
import argparse
import json
from math import comb
from pathlib import Path
import random
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ifdemo"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reproduction/if-paired-comparison.md")
    args = parser.parse_args()
    calls = {}
    for rep in sorted((OUT / "judge").glob("rep*")):
        for path in (rep / "episodes").glob("*.json"):
            judged = json.loads(path.read_text())
            score = (judged.get("scores") or {}).get("official")
            calls.setdefault(path.stem, []).append({"score": score, "verdict": (judged.get("gate") or {}).get("verdict")})
    order = json.loads((OUT / "cases.json").read_text())["order"]
    rows = []
    for task in order:
        arms = {}
        for arm in ("L", "C"):
            scored = [c for c in calls.get(f"{arm}-if-{task}", []) if c["score"] is not None]
            arms[arm] = scored
        if len(arms["L"]) >= 3 and len(arms["C"]) >= 3:
            rows.append({"task": task, "L": st.mean(c["score"] for c in arms["L"]), "C": st.mean(c["score"] for c in arms["C"]),
                         "Ls": [c["score"] for c in arms["L"]], "Cs": [c["score"] for c in arms["C"]],
                         "Lv": [c["verdict"] for c in arms["L"]], "Cv": [c["verdict"] for c in arms["C"]]})
    n = len(rows)
    deltas = [r["C"] - r["L"] for r in rows]
    rng = random.Random(42)
    boot = sorted(st.mean(rng.choice(deltas) for _ in deltas) for _ in range(10000)) if deltas else []
    ci = (boot[249], boot[9749]) if boot else (None, None)
    better, worse = sum(d > 0 for d in deltas), sum(d < 0 for d in deltas)
    m = better + worse
    sign_p = sum(comb(m, k) for k in range(better, m + 1)) / 2 ** m if m else None  # one-sided
    gate = {arm: {v: sum(r[arm + "v"].count(v) for r in rows) for v in ("pass", "partial", "fail", "not_applicable")}
            for arm in ("L", "C")}
    clean = sorted((r for r in rows if min(r["Cs"]) > max(r["Ls"])), key=lambda r: r["L"] - r["C"])
    f = lambda v: "—" if v is None else f"{v:.2f}"
    lines = ["# Paired IF comparison: canonical controller vs baseline player", "",
             f"**{n} of {len(order)} supported IF tasks complete** (both arms, 3 Gemini calls each). Seed 42; each pair on one GPU.",
             "", "## Result", "",
             "| | Baseline | Canonical controller | Δ |", "|---|---:|---:|---:|",
             f"| IF mean | {f(st.mean(r['L'] for r in rows) if rows else None)} | {f(st.mean(r['C'] for r in rows) if rows else None)} | "
             f"{f(st.mean(deltas) if deltas else None)} |",
             f"| Gate pass (calls) | {gate['L']['pass']} | {gate['C']['pass']} | |",
             f"| Gate fail (calls) | {gate['L']['fail']} | {gate['C']['fail']} | |", "",
             f"Paired mean difference {f(st.mean(deltas) if deltas else None)}, task-bootstrap 95% interval [{f(ci[0])}, {f(ci[1])}]; "
             f"{better} better / {n - better - worse} equal / {worse} worse; one-sided sign test p = "
             f"{'—' if sign_p is None else f'{sign_p:.3f}'}.", "",
             "## Per task", "", "| Task | Baseline calls | Ours calls | Δ | Baseline gates | Ours gates |", "|---|---|---|---:|---|---|"]
    for r in rows:
        lines.append(f"| {r['task']} | {', '.join(f'{s:.1f}' for s in r['Ls'])} | {', '.join(f'{s:.1f}' for s in r['Cs'])} | "
                     f"{r['C'] - r['L']:+.2f} | {', '.join(map(str, r['Lv']))} | {', '.join(map(str, r['Cv']))} |")
    lines += ["", "Tasks where every corrected call beats every baseline call (demo candidates): "
              + (", ".join(f"{r['task']} ({r['L']:.2f}→{r['C']:.2f})" for r in clean) or "none") + ".", "",
              "## Setup and caveats", "",
              "- Baseline = frozen full-run configuration (`5b8a5b8`, `--text-prompt both`, 40 chunks, Haiku 4.5 player).",
              "- Canonical controller = `lingbot_released_baseline --generate --validate-controller` on main, 60 chunks, hold-service "
              "token cap 400.",
              "- Compares two complete configurations (episode length, text prompt and controller all differ), not an isolated ablation.",
              "- Judge: `gemini-3.1-pro-preview` on Vertex with the released rubric; the paper's exact model id is unavailable.",
              "- GPUs: Lambda H100 SXM and the local RTX PRO 6000; both arms of a task always ran on the same GPU.", ""]
    args.output.write_text("\n".join(lines))
    print(json.dumps({"n": n, "delta": st.mean(deltas) if deltas else None, "ci": ci, "better": better, "worse": worse,
                      "sign_p": sign_p, "gate": gate, "clean": [r["task"] for r in clean]}))


if __name__ == "__main__":
    main()
