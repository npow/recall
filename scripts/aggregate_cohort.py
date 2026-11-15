"""Aggregate guarded cohort results: paired stock vs 46/6 scores, episode lengths, END reasons, revisits.

Reads outputs/lingbot-reproduction/cohort-*/cohort-results.json, fetches each episode's run.json from the
Modal volume (MODAL_PROFILE=nissan-pow), computes whether the commanded path contains a revisit, and writes
reproduction/cohort/guarded-results.json and a markdown table.
"""
import glob
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/cohort-runs"


def fetch(identifier):
    target = CACHE / identifier
    if not (target / "run.json").exists():
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(["modal", "volume", "get", "worldsync-lingbot-results", f"{identifier}/run.json", str(target)],
                       check=True, env={**os.environ, "MODAL_PROFILE": "nissan-pow"}, capture_output=True)
    path = target / "run.json"
    if not path.exists():
        found = list(target.rglob("run.json"))
        path = found[0]
    return json.loads(path.read_text())


def revisit(run):
    """Commanded-path revisit: leaves the start (>60 deg or >5 units) and later returns (<30 deg and <15% of excursion)."""
    states = []
    for action in run["actions"]:
        for key in ("pose_before", "pose_after"):
            if action.get(key):
                states.append(action[key])
    if not states:
        return {"revisit": False, "absence_s": 0.0}
    left_at, max_excursion, best = None, 0.0, None
    for state in states:
        distance = math.dist(state["position_normalized_units"], (0, 0, 0))
        heading = abs((state["heading_degrees"] + 180) % 360 - 180)
        away = distance > 5 or heading > 60
        max_excursion = max(max_excursion, distance)
        t = state["world_time_seconds"]
        if away and left_at is None:
            left_at = t
        if left_at is not None and not away and distance < max(2.0, 0.15 * max_excursion) and heading < 30:
            best = t - left_at
    return {"revisit": best is not None, "absence_s": best or 0.0}


def main():
    rows = []
    for path in sorted(glob.glob(str(ROOT / "outputs/lingbot-reproduction/cohort-*/cohort-results.json"))):
        for item in json.loads(Path(path).read_text()):
            if not item.get("identifier"):
                continue
            run = fetch(item["identifier"])
            judge = item.get("judge") or {}
            rows.append({"cohort": Path(path).parent.name, "task": item["task"], "arm": item["arm"], "seed": item["seed"],
                         "score": judge.get("reported_score"), "gate": judge.get("gate"),
                         "seconds": run.get("generated_seconds"), "interactions": run.get("agent_interactions"),
                         "termination": run.get("termination"),
                         "end_reason": next((a.get("reason") for a in reversed(run["actions"]) if a["operation"] == "END"), None),
                         "guard_events": len(run.get("end_guard_events", [])), **revisit(run),
                         "error": item.get("error")})
    out = ROOT / "reproduction/cohort"
    out.mkdir(parents=True, exist_ok=True)
    (out / "guarded-results.json").write_text(json.dumps(rows, indent=2) + "\n")
    paired = {}
    for r in rows:
        paired.setdefault((r["task"], r["seed"]), {})[r["arm"]] = r
    lines = ["| Case | Seed | Stock | 46/6 | Stock s / revisit | 46/6 s / revisit | guard rejects (stock/46) |", "| --- | --- | --- | --- | --- | --- | --- |"]
    deltas = {"all": [], "long": []}
    for (task, seed), arms in sorted(paired.items()):
        s, c = arms.get("stock"), arms.get("cache46")
        f = lambda r: "n/a" if not r else f"{r['seconds']:.1f} / {'yes' if r['revisit'] else 'no'} ({r['absence_s']:.1f}s)"
        sc = lambda r: "n/a" if not r or r["score"] is None else str(r["score"])
        lines.append(f"| {task} | {seed} | {sc(s)} | {sc(c)} | {f(s)} | {f(c)} | {s['guard_events'] if s else '-'}/{c['guard_events'] if c else '-'} |")
        if s and c and s["score"] is not None and c["score"] is not None:
            deltas["all"].append(c["score"] - s["score"])
            if s["revisit"] and c["revisit"] and max(s["absence_s"], c["absence_s"]) > 4.5:
                deltas["long"].append(c["score"] - s["score"])
    for name, d in deltas.items():
        if d:
            lines.append(f"\nPaired delta (46/6 - stock), {name}: mean {statistics.mean(d):+.3f}, n={len(d)}, "
                         f"wins {sum(x > 0 for x in d)}, ties {sum(x == 0 for x in d)}, losses {sum(x < 0 for x in d)}")
    for arm in ("stock", "cache46"):
        scores = [r["score"] for r in rows if r["arm"] == arm and r["score"] is not None]
        if scores:
            lines.append(f"\nMean {arm}: {statistics.mean(scores):.3f} (n={len(scores)}); revisit episodes: "
                         f"{sum(r['revisit'] for r in rows if r['arm'] == arm)}/{sum(1 for r in rows if r['arm'] == arm)}")
    (out / "guarded-results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
