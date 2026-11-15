"""Aggregate Gemini-judged cases into the PlayWorld overall: equal mean of the GC, IF, in-sight and out-of-sight dimension means.

    python scripts/aggregate_full.py --label final --output reproduction/full-baseline-gemini.json

Per-case score = mean over judge repeats of the effective score (gate failure -> 1). An overall is reported only when all
171 cases have at least one valid judgement; otherwise coverage is listed and overall is null.
"""
import argparse
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "external/PlayWorld/data"
EXPECTED = {"gc": 48, "if": 50, "insight": 30, "outsight": 43}


def split_of():
    mapping = {}
    for split in EXPECTED:
        for record in json.loads((DATA / split / "data.json").read_text()):
            mapping[record["task_id"]] = split
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="final")
    parser.add_argument("--output", type=Path, default=ROOT / "reproduction/full-baseline-gemini.json")
    args = parser.parse_args()
    data = json.loads((ROOT / f"outputs/rejudge/{args.label}-gemini.json").read_text())
    splits = split_of()
    per_split = {s: {} for s in EXPECTED}
    for task, items in data["cases"].items():
        scores = [i["score"] for i in items if i["score"] is not None]
        if scores and task in splits:
            per_split[splits[task]][task] = {"score": round(st.mean(scores), 3), "judgements": scores,
                                              "gates": [i["gate"] for i in items]}
    dims = {s: (round(st.mean(v["score"] for v in cases.values()), 4) if cases else None) for s, cases in per_split.items()}
    scored = {s: len(c) for s, c in per_split.items()}
    complete = all(scored[s] == EXPECTED[s] for s in EXPECTED)
    out = {"judge": "Gemini 3.1 Pro preview via Vertex (not the paper's API alias)", "protocol": "released prompts/sampling/gate; local LingBot-World2 14B causal-fast; Haiku player; seed 42",
           "judgements_per_video": 2, "expected": EXPECTED, "scored": scored, "coverage_complete": complete,
           "dimension_means_partial": dims, "overall": round(st.mean(dims.values()), 4) if complete else None,
           "gc": dims["gc"], "if": dims["if"], "insight": dims["insight"], "outsight": dims["outsight"],
           "scored_total": sum(scored.values()), "scored_count": sum(scored.values()),
           "note": "overall is null until all 171 cases are scored; dimension means over scored cases are partial", "cases": per_split}
    out["scored"] = sum(scored.values()) if complete else scored
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("scored", "coverage_complete", "dimension_means_partial", "overall")}))


if __name__ == "__main__":
    main()
