"""Step-2 report for memory v2: adaptive revisit episodes against the final1 stock episodes.

Judge-independent metric: DINO(start frame, frame at each commanded return to the start pose).
Judge metric: mean of the Gemini scores per video (files written by scripts/director_score.py-style
scorers are read from --scores-memv2 / --scores-stock: one JSON per episode and call, with
`scores.official`).

Usage: python -m scripts.memv2_step2_report --memv2 outputs/memv2-step2 --stock <final1 dir> --out reproduction/memory-v2/step2
"""
import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from worldsync.lingbot_controls import CameraPlan
from worldsync.lingbot_memv2 import DinoDescriptor, latent_pixel_indices, pose_close


def returns(folder, describe, min_gap_chunks=3):
    """DINO similarity to the start frame at every chunk whose commanded end pose is back at the start."""
    run = json.loads((folder / "run.json").read_text())
    counts = [chunk["frames"] for chunk in run["chunks"]]
    actions = [(a.get("action") or "WAIT") for a in run["actions"]][:len(counts)]
    actions += ["WAIT"] * (len(counts) - len(actions))
    video = iio.imread(folder / "rollout.mp4")
    camera, offsets, total = CameraPlan(), [], 0
    for count in counts:
        offsets.append(total)
        total += count
    away, found = False, []
    start = None
    for chunk, action in enumerate(actions):
        camera.advance(action)
        pose = camera.poses[-1]
        if start is None:
            start = camera.poses[0]
        at_start = pose_close(start, pose, 20, 1.0)
        if not at_start:
            away = True
        elif away and chunk >= min_gap_chunks:
            frame = offsets[chunk] + latent_pixel_indices(chunk, counts[chunk])[-1]
            found.append((chunk, frame))
            away = False
    if not found:
        return run, []
    features = describe([video[0]] + [video[frame] for _, frame in found])
    return run, [{"chunk": chunk, "dino_vs_start": float(features[i + 1] @ features[0])} for i, (chunk, _) in enumerate(found)]


def bootstrap(values, draws=5000, seed=0):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (draws, len(values))).mean(axis=1)
    return [float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


FIRST_CALL_ONLY = False


def judge_scores(root, episode):
    if root is None:
        return []
    scores = []
    pattern = f"rep1/**/{episode}.json" if FIRST_CALL_ONLY else f"**/{episode}.json"
    for path in sorted(Path(root).glob(pattern)):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        value = (data.get("scores") or {}).get("official")
        if value is not None and not data.get("error"):
            scores.append(float(value))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memv2", type=Path, required=True)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, default=Path("reproduction/memory-v2-step2-jobs.jsonl"))
    parser.add_argument("--scores-memv2", type=Path)
    parser.add_argument("--scores-stock", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first-call-only", action="store_true", help="use only the first Gemini call of each video")
    args = parser.parse_args()
    global FIRST_CALL_ONLY
    FIRST_CALL_ONLY = args.first_call_only
    describe = DinoDescriptor()
    rows = []
    for job in map(json.loads, args.jobs.read_text().splitlines()):
        mem_dir, stock_dir = args.memv2 / job["id"], args.stock / job["stock_episode"]
        row = {"case": f"{job['split']}:{job['task_id']}", "split": job["split"], "development": job["development"],
               "arm": job.get("arm", "memv2"), "kv_source": job.get("kv_source"), "task_id": job["task_id"],
               "memv2_video": str((mem_dir / "rollout.mp4").resolve()), "stock_video": str((stock_dir / "rollout.mp4").resolve())}
        if not (mem_dir / "run.json").exists() or not (stock_dir / "run.json").exists():
            row["status"] = "missing"
            rows.append(row)
            continue
        mem_run, mem_returns = returns(mem_dir, describe) if (mem_dir / "rollout.mp4").exists() else (json.loads((mem_dir / "run.json").read_text()), [])
        stock_run, stock_returns = returns(stock_dir, describe)
        memory = mem_run.get("memory") or {}
        trace = memory.get("trace", [])
        row.update({"status": mem_run.get("status"), "error": mem_run.get("error"),
                    "chunks": {"memv2": len(mem_run.get("chunks", [])), "stock": len(stock_run.get("chunks", []))},
                    "fired_chunks": [t["chunk"] for t in trace if t.get("attention_replaced")],
                    "disk_gb": memory.get("disk_bytes_written", 0) / 1e9, "disk_read_s": memory.get("disk_read_seconds"),
                    "elapsed_s": {"memv2": mem_run.get("elapsed_s"), "stock": stock_run.get("elapsed_s")},
                    "peak_gb": {"memv2": (mem_run.get("peak_allocated_bytes") or 0) / 1e9, "stock": (stock_run.get("peak_allocated_bytes") or 0) / 1e9},
                    "returns": {"memv2": mem_returns, "stock": stock_returns},
                    "judge": {"memv2": judge_scores(args.scores_memv2, job["id"]),
                              "stock": judge_scores(args.scores_stock, job["stock_episode"])}})
        rows.append(row)
    reference = [r for r in rows if r.get("status") == "completed" and r["arm"] != "memv2"]
    done = [r for r in rows if r.get("status") == "completed" and r["arm"] == "memv2"]

    def paired(key, subset):
        deltas = []
        for row in subset:
            if key == "dino":
                a, b = row["returns"]["memv2"], row["returns"]["stock"]
                if a and b:
                    deltas.append(np.mean([x["dino_vs_start"] for x in a]) - np.mean([x["dino_vs_start"] for x in b]))
            else:
                a, b = row["judge"]["memv2"], row["judge"]["stock"]
                if a and b:
                    deltas.append(np.mean(a) - np.mean(b))
        return {"pairs": len(deltas), "mean_ci95": bootstrap(deltas),
                "wins_ties_losses": [int(sum(d > 1e-9 for d in deltas)), int(sum(abs(d) <= 1e-9 for d in deltas)), int(sum(d < -1e-9 for d in deltas))]}
    groups = {"all": done, "held_out": [r for r in done if not r["development"]],
              "gc": [r for r in done if r["split"] == "gc"], "outsight": [r for r in done if r["split"] == "outsight"],
              "fired": [r for r in done if r["fired_chunks"]], "never_fired": [r for r in done if not r["fired_chunks"]],
              "reference_arm": reference}
    summary = {"jobs": len(rows), "completed": len(done), "failed": [r["case"] for r in rows if r.get("status") not in ("completed",)],
               "episodes_where_retrieval_fired": sum(bool(r["fired_chunks"]) for r in done),
               "mean_disk_gb": float(np.mean([r["disk_gb"] for r in done])) if done else None,
               "dino_return_delta": {name: paired("dino", group) for name, group in groups.items()},
               "judge_delta": {name: paired("judge", group) for name, group in groups.items()}}
    args.out.mkdir(parents=True, exist_ok=True)
    import csv
    from worldsync import gemini_vertex_judge
    with open(args.out / "per-case.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "task_id", "arm", "development", "objective", "stock_scores", "memv2_scores", "stock_mean",
                         "memv2_mean", "paired_difference", "fired_chunks", "memv2_chunks", "stock_chunks",
                         "dino_return_memv2", "dino_return_stock", "memv2_video", "stock_video"])
        for row in rows:
            if row.get("status") != "completed":
                continue
            task = gemini_vertex_judge.load_dataset_task(Path("external/PlayWorld"), row["split"], row["task_id"])
            a, b = row["judge"]["memv2"], row["judge"]["stock"]
            mean_a, mean_b = (float(np.mean(a)) if a else None), (float(np.mean(b)) if b else None)
            writer.writerow([row["split"], row["task_id"], row["arm"], row["development"], task.get("prompt", ""),
                             " ".join(f"{x:.2f}" for x in b), " ".join(f"{x:.2f}" for x in a),
                             "" if mean_b is None else f"{mean_b:.3f}", "" if mean_a is None else f"{mean_a:.3f}",
                             "" if None in (mean_a, mean_b) else f"{mean_a - mean_b:+.3f}",
                             " ".join(map(str, row["fired_chunks"])), row["chunks"]["memv2"], row["chunks"]["stock"],
                             " ".join(f"{x['dino_vs_start']:.3f}" for x in row["returns"]["memv2"]),
                             " ".join(f"{x['dino_vs_start']:.3f}" for x in row["returns"]["stock"]),
                             row["memv2_video"], row["stock_video"]])
    (args.out / "report.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
