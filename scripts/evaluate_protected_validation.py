"""Evaluate every frozen new-image/new-seed memory comparison that completed."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_protected_recent import fetch, compare


def evaluate(case, judge):
    output = ROOT / "reproduction/protected-recent-validation" / f"{case['task_id']}-seed{case['seed']}"
    labels = ("reference18_read4", "candidate22_read8")
    sources = [ROOT / "outputs/lingbot-reproduction" / f"protected-validation-{case['task_id']}-seed{case['seed']}-{label}.json"
               for label in labels]
    if not all(path.exists() for path in sources):
        return {**case, "status": "generation_pending"}
    for label, source in zip(labels, sources):
        fetch(label, source, output)
    return {**case, **compare(labels, judge, output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", action="store_true")
    args = parser.parse_args()
    plan = json.loads((ROOT / "reproduction/protected-recent-validation-plan.json").read_text())
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda case: evaluate(case, args.judge), plan["cases"]))
    summary = {"scope": plan["scope"], "complete": all(row["status"] == "completed" for row in results),
               "declared_pairs": len(plan["cases"]), "results": results,
               "acceptance": plan["acceptance"], "caveat": "Fixed-path mechanism study, not agent-driven task scores or leaderboard reproduction."}
    output = ROOT / "reproduction/protected-recent-validation"
    output.mkdir(exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
