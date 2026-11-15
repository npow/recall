"""Exploratory robustness checks for the revisit-horizon benchmark (not pre-registered)."""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import revisit_horizon as rh

report = json.loads((rh.REPORT / "analysis.json").read_text())
table = {(e["task_id"], e["horizon_s"], e["arm"]): e for e in report["per_episode"]}
out = {}
for horizon in (0, 6, 12, 24):
    entry = report["horizons"][str(horizon)]
    scenes = entry["scenes"]
    row = {}
    for rep in (0, 1):  # paired difference using one judge call at a time
        deltas = [table[(s, horizon, "memory")]["call_scores"][rep] - table[(s, horizon, "stock")]["call_scores"][rep]
                  for s in scenes]
        row[f"paired_score_call{rep + 1}"] = {"mean": sum(deltas) / len(deltas), "ci95": rh.bootstrap(deltas)}
    row["gate_verdicts"] = {arm: entry[arm]["gate"] for arm in rh.ARMS}
    row["episodes_at_floor_1.0"] = {arm: entry[arm]["at_floor_1.0"] for arm in rh.ARMS}
    row["category_means"] = {arm: {k: round(v["mean"], 2) for k, v in entry["exploratory_category_means"][arm].items()}
                             for arm in rh.ARMS}
    row["dino_away_view_vs_start"] = {arm: round(entry[arm]["dino_away_vs_frame12"]["mean"], 2) for arm in rh.ARMS}
    gaps = [abs(table[(s, horizon, arm)]["call_scores"][0] - table[(s, horizon, arm)]["call_scores"][1])
            for s in scenes for arm in rh.ARMS]
    row["mean_abs_gap_between_two_calls_same_video"] = round(sum(gaps) / len(gaps), 2)
    out[str(horizon)] = row
times, peak = collections.defaultdict(list), collections.defaultdict(list)
for e in report["per_episode"]:
    run = json.loads((rh.RESULTS / e["id"] / "run.json").read_text())
    times[f"{e['arm']}-H{e['horizon_s']}"].append(run["elapsed_s"])
    peak[e["arm"]].append(run["peak_allocated_bytes"] / 1e9)
out["mean_job_seconds"] = {k: round(sum(v) / len(v)) for k, v in sorted(times.items())}
out["max_peak_gpu_gb"] = {k: round(max(v), 1) for k, v in peak.items()}
(rh.REPORT / "robustness.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
