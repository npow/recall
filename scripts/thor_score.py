"""Score finished Thor seed-43 episodes with the Gemini Vertex judge (resumable; run repeatedly or with --watch).

Builds a manifest of episodes that have rollout.mp4 + run.json and are not yet scored, then calls
worldsync.gemini_vertex_judge.run_manifest (gcloud login; tokens stay in memory, never printed).
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worldsync.gemini_vertex_judge import run_manifest  # noqa: E402

OUT = ROOT / "outputs/thor-seed43"
JUDGED = OUT / "gemini"
PLAYWORLD = ROOT / "external/PlayWorld"


def pending():
    rows = []
    for directory in sorted((OUT / "episodes").iterdir()):
        if not directory.is_dir() or ".partial-" in directory.name:
            continue
        if not ((directory / "rollout.mp4").is_file() and (directory / "run.json").is_file()):
            continue
        parts = directory.name.split("-")  # stock-<split>-<task>-seed43
        split, task = parts[1], parts[2]
        done = JUDGED / "episodes" / f"{directory.name}.json"
        if done.is_file() and not json.loads(done.read_text()).get("error"):
            continue
        rows.append({"episode_id": directory.name, "split": split, "task_id": task,
                     "video": str((directory / "rollout.mp4").resolve()),
                     "reference_image": str((PLAYWORLD / "data" / split / "images" / f"{task}.jpg").resolve())})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    while True:
        rows = pending()
        if rows:
            manifest = OUT / "manifest.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            run_manifest(manifest, JUDGED, PLAYWORLD, workers=2)
        if not args.watch:
            return
        if not rows and (OUT / "runner.log").read_text().count('"all_done"'):
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
