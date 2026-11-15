"""Gemini scoring for the director development experiment, three independent judge calls per video.

    python scripts/director_score.py stock          # 2 extra calls for the 20 existing stock videos (call 1 = full run)
    python scripts/director_score.py arms           # fetch finished dev episodes, 3 calls each (re-run to pick up new ones)
    python scripts/director_score.py table          # mean-of-3 per arm and dimension vs stock on the same cases

Each repetition has its own output directory and cache, so calls are independent. Released PlayWorld prompts, rubric,
gate and aggregation via worldsync.gemini_vertex_judge (Vertex, gcloud login; tokens never printed).
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/director-dev"
VOLUME = "worldsync-lingbot-results"
DIMENSION = {"gc": "GC", "if": "IF", "insight": "insight", "outsight": "OE"}
REPS = (1, 2, 3)


def cases():
    return json.loads((ROOT / "outputs/director-dev-cases.json").read_text())


def reference_image(split, task):
    from worldsync import gemini_vertex_judge
    record = gemini_vertex_judge.load_dataset_task(ROOT / "external/PlayWorld", split, task)
    return str((ROOT / "external/PlayWorld/data" / record["image_path"]).resolve())


def judge(rows, directory, workers):
    from worldsync import gemini_vertex_judge
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    gemini_vertex_judge.run_manifest(manifest, directory, playworld=ROOT / "external/PlayWorld", workers=workers)


def stock(args):
    rows = [{"episode_id": f"stock-{c['split']}-{c['task_id']}-seed42", "split": c["split"], "task_id": c["task_id"],
             "video": c["stock_video"], "reference_image": reference_image(c["split"], c["task_id"]),
             "world_model": "LingBot-World2 14B causal-fast"} for c in cases()]
    for rep in (2, 3):
        judge(rows, OUT / "judge/stock" / f"rep{rep}", args.workers)


def fetch(directory, target):
    target.mkdir(parents=True, exist_ok=True)
    for name in ("rollout.mp4", "run.json"):
        if not (target / name).is_file():
            subprocess.run(["modal", "volume", "get", VOLUME, f"{directory}/{name}", str(target) + "/", "--force"],
                           capture_output=True, text=True)
    return (target / "rollout.mp4").is_file() and (target / "run.json").is_file()


def arms(args):
    status = json.loads((ROOT / "outputs/director-dev-status.json").read_text())
    rows = []
    for arm, data in status["arms"].items():
        for row in data["completed"]:
            identifier = f"{arm}-{row['split']}-{row['task_id']}-seed{row['seed']}"
            target = OUT / "episodes" / identifier
            if fetch(row["result_directory"], target):
                rows.append({"episode_id": identifier, "split": row["split"], "task_id": row["task_id"],
                             "video": str((target / "rollout.mp4").resolve()),
                             "reference_image": reference_image(row["split"], row["task_id"]),
                             "world_model": "LingBot-World2 14B causal-fast"})
    print(json.dumps({"fetched": len(rows)}), flush=True)
    for rep in REPS:
        judge(rows, OUT / "judge/arms" / f"rep{rep}", args.workers)


def lambda_episodes(args):
    """Score episodes synced back from the Lambda boxes: outputs/director-lambda/<id>/{rollout.mp4,run.json}."""
    rows = []
    for directory in sorted((ROOT / "outputs/director-lambda").glob("*-seed42")):
        run, video = directory / "run.json", directory / "rollout.mp4"
        if not (run.is_file() and video.is_file()) or json.loads(run.read_text()).get("status") != "completed":
            continue
        arm, split, task = directory.name.split("-")[:3]
        rows.append({"episode_id": directory.name, "split": split, "task_id": task, "video": str(video.resolve()),
                     "reference_image": reference_image(split, task), "world_model": "LingBot-World2 14B causal-fast"})
    print(json.dumps({"lambda_episodes": len(rows)}), flush=True)
    for rep in REPS:
        judge(rows, OUT / "judge/lambda" / f"rep{rep}", args.workers)


def official(path):
    if not path.is_file():
        return None, None
    result = json.loads(path.read_text())
    return (result.get("scores") or {}).get("official"), (result.get("gate") or {}).get("verdict", (result.get("gate") or {}).get("instruction_following"))


def collect():
    """{arm: {(split, task): [scores...]}} with stock call 1 taken from the full run."""
    table = {}
    full = Path(cases()[0]["stock_video"]).parents[2] / "gemini/episodes"
    for case in cases():
        key = (case["split"], case["task_id"])
        name = f"stock-{case['split']}-{case['task_id']}-seed42.json"
        scores = [official(full / name)[0]] + [official(OUT / "judge/stock" / f"rep{rep}" / "episodes" / name)[0] for rep in (2, 3)]
        table.setdefault("stock", {})[key] = [s for s in scores if s is not None]
    for source, prefix in (("arms", "modal:"), ("lambda", "")):
        for rep in REPS:
            directory = OUT / "judge" / source / f"rep{rep}" / "episodes"
            for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
                arm, split, task = path.stem.split("-")[:3]
                score = official(path)[0]
                if score is not None:
                    table.setdefault(prefix + arm, {}).setdefault((split, task), []).append(float(score))
    return table


def bootstrap_ci(values, draws=4000, seed=7):
    """Percentile bootstrap over cases for the mean paired difference."""
    import random
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return [round(means[int(0.025 * draws)], 3), round(means[int(0.975 * draws) - 1], 3)]


def table(args):
    data = collect()
    h200_stock = {key: statistics.mean(values) for key, values in data["stock"].items() if values}
    replay = {key: statistics.mean(values) for key, values in data.get("replay", {}).items() if values}
    report = {"per_case": {}, "arms": {}, "controls": "Lambda arms are paired with the Lambda `replay` (stock text, same GPU model); "
              "`modal:` arms and `replay` itself are paired with the existing H200 stock full-run video."}
    for arm, rows in data.items():
        for key, values in rows.items():
            report["per_case"].setdefault(f"{key[0]}:{key[1]}", {})[arm] = {"calls": values, "mean": statistics.mean(values) if values else None}
        if arm == "stock":
            continue
        stock_means = h200_stock if arm.startswith("modal:") or arm == "replay" else replay
        summary = {}
        for split in ("if", "outsight", "insight", "gc"):
            pairs = [(statistics.mean(values), stock_means[key]) for key, values in rows.items()
                     if key[0] == split and values and key in stock_means]
            if pairs:
                summary[DIMENSION[split]] = {"n": len(pairs), "arm_mean_of_3": round(statistics.mean(p[0] for p in pairs), 3),
                                             "control_mean_of_3": round(statistics.mean(p[1] for p in pairs), 3),
                                             "delta": round(statistics.mean(p[0] - p[1] for p in pairs), 3),
                                             "delta_ci95": bootstrap_ci([p[0] - p[1] for p in pairs]),
                                             "wins_ties_losses": [sum(p[0] > p[1] + 1e-9 for p in pairs),
                                                                  sum(abs(p[0] - p[1]) <= 1e-9 for p in pairs),
                                                                  sum(p[0] < p[1] - 1e-9 for p in pairs)]}
        report["arms"][arm] = summary
    sd = [statistics.pstdev(values) for rows in data.values() for values in rows.values() if len(values) == 3]
    report["judge_noise"] = {"videos_with_3_calls": len(sd), "mean_within_video_sd": round(statistics.mean(sd), 3) if sd else None}
    (OUT / "table.json").write_text(json.dumps(report, indent=1) + "\n")
    for arm, summary in report["arms"].items():
        print(arm)
        for dimension, row in summary.items():
            print(f"   {dimension:8s} n={row['n']}  arm {row['arm_mean_of_3']:.2f}  control {row['control_mean_of_3']:.2f}  "
                  f"delta {row['delta']:+.2f} {row['delta_ci95']}  W/T/L {row['wins_ties_losses']}")
    print("judge noise", report["judge_noise"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["stock", "arms", "lambda", "table"])
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"stock": stock, "arms": arms, "lambda": lambda_episodes, "table": table}[args.command](args)


if __name__ == "__main__":
    main()
