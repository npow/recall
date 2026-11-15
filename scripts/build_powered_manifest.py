"""Build the Gemini rescoring manifest from downloaded cohort episode directories.

Directory names look like cohort-guard-<arm>-<split>-<TASK>-seed<N>-<timestamp>.
Writes manifest.jsonl for worldsync.gemini_vertex_judge and meta.json with the arm,
seed and our Claude (Opus) judge outcome for each episode.
"""
import argparse
import json
from pathlib import Path
import re

PATTERN = re.compile(r"^powered-(?P<arm>stock|budget|cache46)-(?P<split>gc|outsight|insight|if)-(?P<task>[A-Z]+\d+)-seed(?P<seed>\d+)-\d+$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--playworld", type=Path, default=Path("external/PlayWorld"))
    parser.add_argument("--v3-video", type=Path)
    args = parser.parse_args()
    rows, meta = [], {}
    for directory in sorted(args.episodes.iterdir()):
        match = PATTERN.match(directory.name)
        video = directory / "rollout.mp4"
        if not match or not video.is_file():
            continue
        split, task = match["split"], match["task"]
        opus = {}
        judge_file = directory / "judge.json"
        if judge_file.is_file():
            judge = json.loads(judge_file.read_text())
            gate = judge.get("gate") or {}
            opus = {"verdict": gate.get("verdict"), "score": judge.get("reported_score")}
        episode_id = directory.name
        rows.append({"episode_id": episode_id, "split": split, "task_id": task, "video": str(video.resolve()),
                     "reference_image": str((args.playworld / "data" / split / "images" / f"{task}.jpg").resolve())})
        meta[episode_id] = {"arm": match["arm"], "seed": int(match["seed"]), "task": task, "split": split, "opus": opus}
    if args.v3_video and args.v3_video.is_file():
        rows.append({"episode_id": "v3-stock-GC007", "split": "gc", "task_id": "GC007", "video": str(args.v3_video.resolve()),
                     "reference_image": str((args.playworld / "data/gc/images/GC007.jpg").resolve())})
        meta["v3-stock-GC007"] = {"arm": "v3stock", "seed": 42, "task": "GC007", "split": "gc", "opus": {"verdict": "partial", "score": 3.2}}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (args.output / "meta.json").write_text(json.dumps(meta, indent=2))
    counts = {}
    for item in meta.values():
        counts[item["arm"]] = counts.get(item["arm"], 0) + 1
    print(len(rows), "episodes", counts)


if __name__ == "__main__":
    main()
