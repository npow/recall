"""Fetch finished four-dimension episodes from Modal, build the Gemini manifest, and score what is new.

Resumable: downloads, manifests and Gemini results are all cached, so run it repeatedly while generation runs.

  MODAL_PROFILE=nissan-pow .venv/bin/python scripts/sync_fourdim.py SEED_DIR [--gemini] [--workers 4]

SEED_DIR is outputs/lingbot-reproduction/fourdim-seed42 (it holds generated.json written by
scripts/modal_lingbot_fourdim.py). Episodes land in SEED_DIR/episodes/<identifier>/, manifest and
meta in SEED_DIR/gemini/, per-episode Gemini JSON in SEED_DIR/gemini/episodes/.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_cohort_videos import fetch  # noqa: E402

PATTERN = re.compile(r"^fourdim-(?P<arm>[a-z0-9]+)-(?P<split>gc|outsight|insight|if)-(?P<task>[A-Z]+\d+)-seed(?P<seed>\d+)-\d+$")
PLAYWORLD = ROOT / "external/PlayWorld"


def build_manifest(episodes_dir, gemini_dir):
    rows, meta = [], {}
    images = {}
    for directory in sorted(Path(episodes_dir).iterdir()):
        match = PATTERN.match(directory.name)
        video = directory / "rollout.mp4"
        if not match or not video.is_file():
            continue
        split, task = match["split"], match["task"]
        if split not in images:
            images[split] = {r["task_id"]: r["image_path"] for r in json.loads((PLAYWORLD / "data" / split / "data.json").read_text())}
        opus = {}
        judge_file = directory / "judge.json"
        if judge_file.is_file():
            judge = json.loads(judge_file.read_text())
            gate = judge.get("gate") or {}
            opus = {"verdict": gate.get("verdict"), "score": judge.get("reported_score")}
        rows.append({"episode_id": directory.name, "split": split, "task_id": task, "video": str(video.resolve()),
                     "reference_image": str((PLAYWORLD / "data" / images[split][task]).resolve())})
        meta[directory.name] = {"arm": match["arm"], "seed": int(match["seed"]), "task": task, "split": split, "opus": opus}
    Path(gemini_dir).mkdir(parents=True, exist_ok=True)
    (Path(gemini_dir) / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (Path(gemini_dir) / "meta.json").write_text(json.dumps(meta, indent=2))
    return rows, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed_dir", type=Path)
    parser.add_argument("--gemini", action="store_true", help="score new episodes with the Gemini Vertex judge")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    generated = json.loads((args.seed_dir / "generated.json").read_text())
    directories = [g["identifier"] for g in generated if g.get("identifier") and g.get("returncode") == 0]
    episodes = args.seed_dir / "episodes"
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda d: fetch(d, episodes), directories))
    rows, meta = build_manifest(episodes, args.seed_dir / "gemini")
    print(f"generated ok {len(directories)} / entries {len(generated)}; videos present {len(rows)}", flush=True)
    if args.gemini and rows:
        subprocess.run([sys.executable, "-m", "worldsync.gemini_vertex_judge", "--manifest", str(args.seed_dir / "gemini/manifest.jsonl"),
                        "--output-dir", str(args.seed_dir / "gemini"), "--workers", str(args.workers)], cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
