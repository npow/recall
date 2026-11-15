"""Judge-repeat sensitivity analysis: re-judge floor / gate-affected episodes with 2 extra independent Gemini calls.

    python scripts/judge_repeat.py run --label full1 --phase oe_floor --workers 12
    python scripts/judge_repeat.py run --label full1 --phase rest --workers 12
    python scripts/judge_repeat.py report --label full1

Same released prompts, rubric, gate, sampling and model as the primary run; no changes. Each repeat uses its own
cache directory, so the extra calls are genuinely independent samples (temperature 0.1 as in the primary run).
The frozen single-call results stay the primary numbers; this is a labelled sensitivity analysis.

Selection (out-of-sight and GC episodes, both arms): score exactly 1.0, or gate verdict fail / partial.
Phase `oe_floor` = out-of-sight episodes at exactly 1.0 (the ~66 floor episodes); phase `rest` = everything else selected.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPLITS_SELECTED = ("outsight", "gc")


def load_primary(label):
    rows = []
    with open(ROOT / "reproduction/full-run/per-episode.csv") as handle:
        for row in csv.DictReader(handle):
            if row["error"] == "True" or row["score"] in ("", "None"):
                continue
            row["score"] = float(row["score"])
            row["id"] = f"{row['arm']}-{row['split']}-{row['task']}-seed42"
            rows.append(row)
    return rows


def selected(rows, phase):
    pick = [r for r in rows if r["split"] in SPLITS_SELECTED and (r["score"] == 1.0 or r["gate"] in ("fail", "partial"))]
    floor = [r for r in pick if r["split"] == "outsight" and r["score"] == 1.0]
    rest = [r for r in pick if r not in floor]
    return {"oe_floor": floor, "rest": rest, "all": floor + rest}[phase]


def run(args):
    from worldsync import gemini_vertex_judge
    base = ROOT / "outputs/adaptive" / args.label
    rows = selected(load_primary(args.label), args.phase)
    manifest_rows = []
    for r in rows:
        video = base / "episodes" / r["id"] / "rollout.mp4"
        if not video.is_file():
            print("missing video", r["id"])
            continue
        task = gemini_vertex_judge.load_dataset_task(ROOT / "external/PlayWorld", r["split"], r["task"])
        manifest_rows.append({"episode_id": r["id"], "split": r["split"], "task_id": r["task"], "video": str(video.resolve()),
                              "reference_image": str((ROOT / "external/PlayWorld/data" / task["image_path"]).resolve()),
                              "world_model": "LingBot-World2 14B causal-fast"})  # neutral: never reveals the arm
    out = base / "judge_repeat"
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / f"manifest-{args.phase}.jsonl"
    manifest.write_text("".join(json.dumps(m) + "\n" for m in manifest_rows))
    print(json.dumps({"phase": args.phase, "episodes": len(manifest_rows), "repeats": 2}), flush=True)
    for repeat in (1, 2):
        gemini_vertex_judge.run_manifest(manifest, out / f"rep{repeat}", playworld=ROOT / "external/PlayWorld",
                                         workers=args.workers)
        print(json.dumps({"event": "repeat_done", "repeat": repeat}), flush=True)


def gate_of(result):
    return (result.get("gate") or {}).get("verdict")


def score_of(result):
    return (result.get("scores") or {}).get("official")


def collect(label):
    base = ROOT / "outputs/adaptive" / label
    primary = {r["id"]: r for r in load_primary(label)}
    rows = []
    for identifier, p in primary.items():
        reps = []
        for repeat in (1, 2):
            path = base / "judge_repeat" / f"rep{repeat}/episodes/{identifier}.json"
            if path.is_file():
                result = json.loads(path.read_text())
                if not result.get("error") and score_of(result) is not None:
                    reps.append({"score": float(score_of(result)), "gate": gate_of(result)})
        if len(reps) == 2:
            scores = [p["score"]] + [r["score"] for r in reps]
            gates = [p["gate"]] + [r["gate"] for r in reps]
            majority = max(set(gates), key=lambda g: (gates.count(g), g == p["gate"]))
            rows.append({**p, "rep_scores": [r["score"] for r in reps], "rep_gates": [r["gate"] for r in reps],
                         "mean3": sum(scores) / 3, "sd3": statistics.pstdev(scores), "majority_gate": majority,
                         "gate_flip": majority != p["gate"], "score_flip_from_1": p["score"] == 1.0 and sum(scores) / 3 > 1.0 + 1e-9})
    return primary, rows


def dimension_means(primary_scores, override):
    """overall = equal mean of per-dimension means; `override` maps id -> replacement score."""
    from worldsync import fourdim
    out = {}
    for arm in ("stock", "budget"):
        scores = {}
        for identifier, p in primary_scores.items():
            if p["arm"] == arm:
                scores[(p["split"], p["task"])] = override.get(identifier, p["score"])
        dims = fourdim.dimension_means(scores)
        out[arm] = {"dims": dims, "overall": fourdim.overall(dims), "scores": scores}
    delta = fourdim.bootstrap_paired_delta(out["budget"]["scores"], out["stock"]["scores"])
    return out, delta


def report(args):
    primary, rows = collect(args.label)
    by_arm = {}
    for arm in ("stock", "budget"):
        sub = [r for r in rows if r["arm"] == arm]
        by_arm[arm] = {"n": len(sub),
                       "single_call_mean": sum(r["score"] for r in sub) / len(sub) if sub else None,
                       "mean_of_3": sum(r["mean3"] for r in sub) / len(sub) if sub else None,
                       "mean_sd": sum(r["sd3"] for r in sub) / len(sub) if sub else None,
                       "gate_flips": sum(r["gate_flip"] for r in sub),
                       "floor_episodes_above_1.0_in_mean": sum(r["score_flip_from_1"] for r in sub)}
    base, base_delta = dimension_means(primary, {})
    override = {r["id"]: r["mean3"] for r in rows}
    rep, rep_delta = dimension_means(primary, override)
    summary = {"episodes_with_two_repeats": len(rows), "by_arm_on_repeated_set": by_arm,
               "single_call_primary": {a: {"dims": base[a]["dims"], "overall": base[a]["overall"]} for a in base},
               "repeated_mean_of_3_replacing_single_call_on_repeated_episodes": {a: {"dims": rep[a]["dims"], "overall": rep[a]["overall"]} for a in rep},
               "paired_budget_minus_stock_single_call": base_delta, "paired_budget_minus_stock_with_repeats": rep_delta,
               "note": "repeated set = out-of-sight and GC episodes at score 1.0 or gate fail/partial; other episodes keep their single call"}
    out = ROOT / "reproduction/judge-repeat"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (out / "per-episode.csv").write_text("id,split,arm,single_score,rep1,rep2,mean3,sd3,gate_single,gate_rep1,gate_rep2,gate_majority,gate_flip\n" + "\n".join(
        f"{r['id']},{r['split']},{r['arm']},{r['score']},{r['rep_scores'][0]},{r['rep_scores'][1]},{r['mean3']:.3f},{r['sd3']:.3f},"
        f"{r['gate']},{r['rep_gates'][0]},{r['rep_gates'][1]},{r['majority_gate']},{r['gate_flip']}" for r in rows) + "\n")
    print(json.dumps(summary, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", default="full1")
    r.add_argument("--phase", choices=("oe_floor", "rest", "all"), default="oe_floor")
    r.add_argument("--workers", type=int, default=12)
    r.set_defaults(func=run)
    s = sub.add_parser("report")
    s.add_argument("--label", default="full1")
    s.set_defaults(func=report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
