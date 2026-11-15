"""Write reproduction/corrected-controller-validation.md from the 12-case legacy vs canonical-controller run.

    python scripts/ccv_report.py --run outputs/ccv --selection outputs/ccv_selection.json

Per video: mean of the three independent Gemini calls (released `scores.official`; a judge call without a score is
reported, never imputed). Per arm: dimension means over cases and the equal-weight overall. Paired difference
(canonical minus legacy) per case, with a case-bootstrap 95% interval. Gate pass counts over all judge calls.
"""
import argparse
import csv
import json
from pathlib import Path
import random
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("gc", "if", "insight", "outsight")
NAMES = {"gc": "GC", "if": "IF", "insight": "In-sight", "outsight": "Out-of-sight"}


def load(run):
    data = {}
    for rep in sorted((run / "judge").glob("rep*")):
        for path in (rep / "episodes").glob("*.json"):
            judged = json.loads(path.read_text())
            score = (judged.get("scores") or {}).get("official")
            verdict = (judged.get("gate") or {}).get("verdict")
            data.setdefault(path.stem, []).append({"score": score, "verdict": verdict})
    return data


def generation(run, arm, split, task):
    if arm == "L":
        path = run / "L" / f"{split}-{task}-attempt1" / "run.json"
    else:
        path = run / "C" / f"{split}-{task}" / "episode" / "run.json"
    try:
        report = json.loads(path.read_text())
        return {"status": report.get("status"), "seconds": report.get("generated_seconds"),
                "termination": report.get("termination"), "player_calls": report.get("agent_interactions")}
    except Exception:
        return {"status": "missing"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/ccv")
    parser.add_argument("--selection", type=Path, default=ROOT / "outputs/ccv_selection.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reproduction/corrected-controller-validation.md")
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    judged = load(args.run)
    frozen = {}
    with open(ROOT / "reproduction/full-run/per-episode.csv") as handle:
        for row in csv.DictReader(handle):
            if row["arm"] == "stock":
                frozen[(row["split"], row["task"])] = row
    rows, missing = [], []
    for split in SPLITS:
        for task in selection["picked"][split]:
            entry = {"split": split, "task": task}
            for arm in ("L", "C"):
                calls = judged.get(f"{arm}-{split}-{task}", [])
                scores = [c["score"] for c in calls if c["score"] is not None]
                entry[arm] = round(st.mean(scores), 3) if scores else None
                entry[arm + "_calls"] = f"{len(scores)}/{len(calls)}"
                entry[arm + "_verdicts"] = [c["verdict"] for c in calls]
                entry[arm + "_gen"] = generation(args.run, arm, split, task)
                if len(scores) < 3:
                    missing.append(f"{arm}-{split}-{task}: {len(scores)} scored of {len(calls)} calls")
            reference = frozen.get((split, task))
            entry["frozen"] = float(reference["score"]) if reference and reference["score"] not in ("", "None") else None
            rows.append(entry)

    def dim_means(arm):
        return {s: (st.mean(r[arm] for r in rows if r["split"] == s and r[arm] is not None)
                    if any(r["split"] == s and r[arm] is not None for r in rows) else None) for s in SPLITS}

    means = {arm: dim_means(arm) for arm in ("L", "C")}
    overall = {arm: (st.mean(means[arm].values()) if all(v is not None for v in means[arm].values()) else None) for arm in means}
    pairs = [r for r in rows if r["L"] is not None and r["C"] is not None]
    deltas = [r["C"] - r["L"] for r in pairs]
    rng = random.Random(42)
    boot = sorted(st.mean(rng.choice(deltas) for _ in deltas) for _ in range(10000)) if deltas else []
    ci = (boot[249], boot[9749]) if boot else (None, None)

    def gates(arm):
        verdicts = [v for r in rows for v in r[arm + "_verdicts"]]
        return {k: verdicts.count(k) for k in ("pass", "partial", "fail", "not_applicable")}, len(verdicts)

    fmt = lambda v: "—" if v is None else f"{v:.2f}"
    lines = ["# Corrected-controller validation: canonical controller vs legacy player (12 cases)", "",
        "**Scope: validation on a 12-case sample, not a benchmark score.** One seed (42), one GPU, three Gemini calls per video.", "",
        "## Setup", "",
        f"- **Cases:** {selection['rule']}: " + ", ".join(f"{NAMES[s]} {' '.join(selection['picked'][s])}" for s in SPLITS) + ".",
        "- **Legacy arm:** frozen full-run config at `5b8a5b8` (`reproduction/full-run-config.json`): `--player-retries 2 "
        "--text-prompt both --offload-text-encoder`, 40 chunks; insight `--text-prompt caption` at 60 chunks; Haiku 4.5 player.",
        "- **Corrected arm:** Codex's canonical controller on current `main` via `--validate-controller` (task objective as text, "
        "canonical timeline, released hold service, 60-chunk budget). Hold-service token cap raised from the released 60 to 400 "
        "(`WORLDSYNC_HOLD_MAX_TOKENS`) because Haiku's replies exceed 60 tokens and every hold check otherwise fails.",
        "- **Hardware:** both arms on the same local RTX PRO 6000 Blackwell (shared with other projects). Not bit-comparable to the Modal frozen run.",
        "- **Player transport:** Haiku 4.5 (`claude-haiku-4-5-20251001`) through the local CLIProxyAPI; the project API key is spend-capped.",
        "- **Judge:** `worldsync/gemini_vertex_judge.py` (released rubric, gate and aggregation), `gemini-3.1-pro-preview` on Vertex, "
        "three independent calls per video (separate caches).", "",
        "## Results", "",
        "| Dimension | Legacy | Corrected | Corrected − legacy |", "|---|---:|---:|---:|"]
    for s in SPLITS:
        a, b = means["L"][s], means["C"][s]
        lines.append(f"| {NAMES[s]} (n=3) | {fmt(a)} | {fmt(b)} | {fmt(None if a is None or b is None else b - a)} |")
    lines.append(f"| **Overall (equal mean of dimensions)** | **{fmt(overall['L'])}** | **{fmt(overall['C'])}** | "
                 f"**{fmt(None if overall['L'] is None or overall['C'] is None else overall['C'] - overall['L'])}** |")
    lines += ["", f"Paired per-case difference (corrected − legacy): mean {fmt(st.mean(deltas) if deltas else None)} over {len(deltas)} pairs, "
              f"case-bootstrap 95% interval [{fmt(ci[0])}, {fmt(ci[1])}]; "
              f"{sum(d > 0 for d in deltas)} better / {sum(d == 0 for d in deltas)} equal / {sum(d < 0 for d in deltas)} worse.", ""]
    for arm, name in (("L", "Legacy"), ("C", "Corrected")):
        counts, total = gates(arm)
        lines.append(f"- **Gate verdicts, {name}:** pass {counts['pass']}, partial {counts['partial']}, fail {counts['fail']}, "
                     f"not applicable {counts['not_applicable']} (of {total} judge calls).")
    lines += ["", "## Per case", "",
        "| Case | Legacy | Corrected | Δ | Legacy gates | Corrected gates | Frozen stock (Modal H200, 1 call) |",
        "|---|---:|---:|---:|---|---|---:|"]
    for r in rows:
        d = None if r["L"] is None or r["C"] is None else r["C"] - r["L"]
        lines.append(f"| {r['split']}:{r['task']} | {fmt(r['L'])} ({r['L_calls']}) | {fmt(r['C'])} ({r['C_calls']}) | {fmt(d)} | "
                     f"{', '.join(map(str, r['L_verdicts']))} | {', '.join(map(str, r['C_verdicts']))} | {fmt(r['frozen'])} |")
    lines += ["", "## Generation", "", "| Case | Legacy (s, termination, player calls) | Corrected (s, termination, player calls) |", "|---|---|---|"]
    for r in rows:
        g, h = r["L_gen"], r["C_gen"]
        lines.append(f"| {r['split']}:{r['task']} | {g.get('seconds')}, {g.get('termination')}, {g.get('player_calls')} | "
                     f"{h.get('seconds')}, {h.get('termination')}, {h.get('player_calls')} |")
    lines += ["", "## Caveats", "",
        "- Twelve cases and one seed: a single episode varies by about ±1 point between seeds, so only large differences are informative.",
        "- The arms differ in more than the controller: text (caption+objective vs objective only; insight caption vs \"wait and observe\") "
        "and chunk budget (40 vs 60 for moving cases). This compares the two complete configurations as specified.",
        "- Hold-service token cap raised (above). Proxy transport for the player.",
        "- Missing judgements: " + ("none." if not missing else "; ".join(missing) + ". Not imputed."), ""]
    args.output.write_text("\n".join(lines))
    print(json.dumps({"overall": overall, "dims": means, "paired_mean": st.mean(deltas) if deltas else None, "ci": ci, "missing": missing}))


if __name__ == "__main__":
    main()
