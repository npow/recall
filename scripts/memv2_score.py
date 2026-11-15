"""Gemini scoring for memory v2 step 2: three independent calls per video, both arms.

    python scripts/memv2_score.py --memv2 outputs/memv2-step2 --stock <final1 results dir> --out outputs/memv2-step2-judge

Released PlayWorld prompts, rubric, gate and aggregation via worldsync.gemini_vertex_judge
(Vertex, gcloud login; tokens never printed). Each repetition has its own directory and cache.
Re-run to pick up newly finished episodes; finished judgments are kept.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def rows(root, names):
    from worldsync import gemini_vertex_judge
    found = []
    for name in names:
        folder = root / name
        run, video = folder / "run.json", folder / "rollout.mp4"
        if not (run.is_file() and video.is_file()) or json.loads(run.read_text()).get("status") != "completed":
            continue
        _, split, task = name.split("-")[:3]
        record = gemini_vertex_judge.load_dataset_task(ROOT / "external/PlayWorld", split, task)
        found.append({"episode_id": name, "split": split, "task_id": task, "video": str(video.resolve()),
                      "reference_image": str((ROOT / "external/PlayWorld/data" / record["image_path"]).resolve()),
                      "world_model": "LingBot-World2 14B causal-fast"})
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memv2", type=Path, required=True)
    parser.add_argument("--stock", type=Path, help="final1 results dir; omit to reuse existing stock judgments")
    parser.add_argument("--jobs", type=Path, default=ROOT / "reproduction/memory-v2-step2-jobs.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()
    from worldsync import gemini_vertex_judge
    jobs = [json.loads(line) for line in args.jobs.read_text().splitlines() if line.strip()]
    memory = rows(args.memv2, [job["id"] for job in jobs])
    done = {row["task_id"] for row in memory}
    stock = rows(args.stock, sorted({job["stock_episode"] for job in jobs if job["task_id"] in done})) if args.stock else []
    print(json.dumps({"memv2_episodes": len(memory), "stock_episodes": len(stock)}), flush=True)
    for rep in range(1, args.reps + 1):
        for arm, items in (("memv2", memory), ("stock", stock)):
            directory = args.out / arm / f"rep{rep}"
            directory.mkdir(parents=True, exist_ok=True)
            pending = [row for row in items if not (directory / "episodes" / f"{row['episode_id']}.json").is_file()]
            if not pending:
                continue
            manifest = directory / "manifest.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in pending))
            gemini_vertex_judge.run_manifest(manifest, directory, playworld=ROOT / "external/PlayWorld", workers=args.workers)


if __name__ == "__main__":
    main()
