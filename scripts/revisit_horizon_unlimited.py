"""Score and analyse the unlimited-archive arm at 24 s away (see reproduction/revisit-horizon-unlimited.md).

    python scripts/revisit_horizon_unlimited.py score [--watch]   # judge (2 Gemini calls) + DINO for finished videos
    python scripts/revisit_horizon_unlimited.py analyze           # paired comparison vs the existing 24-frame memory and stock

Reuses the released-judge code and helpers of scripts/revisit_horizon.py with the job list swapped; the stock and 24-frame
memory scores come from the committed reproduction/revisit-horizon/analysis.json (not regenerated).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revisit_horizon as rh  # noqa: E402

ROOT = rh.ROOT
rh.JOBS = ROOT / "reproduction/revisit-horizon-unlimited-jobs.jsonl"
OUT = ROOT / "reproduction/revisit-horizon-unlimited"


def analyze(_):
    rows, done, table = rh.load_scores()
    old = json.loads((ROOT / "reproduction/revisit-horizon/analysis.json").read_text())["per_episode"]
    base = {(r["task_id"], r["arm"]): r for r in old if r["horizon_s"] == 24}
    order = [json.loads(line)["task_id"] for line in rh.JOBS.read_text().splitlines() if line.strip()]
    scored = [task for task in order if (task, 24, "memory_unlimited") in table]
    report = {"order_of_preregistered_job_list": order, "generated": len(done), "scored_scenes": scored, "n": len(scored)}
    for label, key in (("vs_24frame_memory", "memory"), ("vs_stock", "stock")):
        deltas = [table[(t, 24, "memory_unlimited")]["score"] - base[(t, key)]["score"] for t in scored]
        report[label] = {**rh.summarize(deltas), "wins": sum(d > 1e-9 for d in deltas),
                         "ties": sum(abs(d) <= 1e-9 for d in deltas), "losses": sum(d < -1e-9 for d in deltas)}
        dd = [table[(t, 24, "memory_unlimited")]["end_vs_frame12"] - base[(t, key)]["end_vs_frame12"] for t in scored]
        report[label]["dino_end_vs_frame12"] = rh.summarize(dd)
    for name, getter in (("unlimited", lambda t: table[(t, 24, "memory_unlimited")]["score"]),
                         ("memory_24frame", lambda t: base[(t, "memory")]["score"]),
                         ("stock", lambda t: base[(t, "stock")]["score"])):
        report[f"mean_{name}"] = rh.summarize([getter(t) for t in scored])
    report["per_scene"] = [{"task_id": t, "unlimited": table[(t, 24, "memory_unlimited")]["score"],
                            "call_scores": table[(t, 24, "memory_unlimited")]["call_scores"],
                            "memory_24frame": base[(t, "memory")]["score"], "stock": base[(t, "stock")]["score"],
                            "dino_unlimited": table[(t, 24, "memory_unlimited")]["end_vs_frame12"],
                            "dino_24frame": base[(t, "memory")]["end_vs_frame12"]} for t in scored]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_scene"}, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["score", "analyze"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--interval", type=int, default=45)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    {"score": rh.score, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()
