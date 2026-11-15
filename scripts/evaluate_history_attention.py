"""Blind complete-video evaluation of every predeclared history-output gate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_protected_recent import fetch, compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", action="store_true")
    args = parser.parse_args()
    plan = json.loads((ROOT / "reproduction/separate-history-attention-plan.json").read_text())
    output = ROOT / "reproduction/separate-history-attention"
    manifests = {}
    for variant in plan["variants"]:
        label = variant["label"]
        source = ROOT / "outputs/lingbot-reproduction" / f"history-attention-{label}.json"
        if source.exists():
            manifests[label] = fetch(label, source, output)
    if "joint" not in manifests:
        raise ValueError("Complete joint-softmax control required")
    pairs = [("joint", variant["label"]) for variant in plan["variants"][1:] if variant["label"] in manifests]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda pair: compare(pair, args.judge, output), pairs))
    summary = {"scope": plan["scope"], "actual_workspace": manifests["joint"].get("workspace"),
               "complete": len(manifests) == len(plan["variants"]) and len(results) == len(plan["variants"]) - 1,
               "results": results, "declared_variants": plan["variants"],
               "caveat": "Untrained separately normalized outputs; development-only. No held-out or leaderboard improvement claim."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
