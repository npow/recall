"""Download rollout.mp4, run.json and judge.json for episode directories in the Modal results volume.

Usage: MODAL_PROFILE=nissan-pow python scripts/fetch_cohort_videos.py DIRS_FILE OUTPUT_DIR [--workers 8]
DIRS_FILE lists one results-volume directory name per line.
"""
import argparse
import concurrent.futures
from pathlib import Path
import subprocess

VOLUME = "worldsync-lingbot-results"
FILES = ("rollout.mp4", "run.json", "judge.json")


def fetch(directory, output):
    target = Path(output) / directory
    target.mkdir(parents=True, exist_ok=True)
    fetched = []
    for name in FILES:
        if (target / name).is_file():
            fetched.append(name)
            continue
        outcome = subprocess.run(["modal", "volume", "get", VOLUME, f"{directory}/{name}", str(target) + "/", "--force"],
                                 capture_output=True, text=True)
        if outcome.returncode == 0 and (target / name).is_file():
            fetched.append(name)
    return directory, fetched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs_file", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    directories = [line.strip() for line in args.dirs_file.read_text().splitlines() if line.strip()]
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda d: fetch(d, args.output), directories))
    missing = [d for d, files in results if "rollout.mp4" not in files]
    print(f"fetched {len(results) - len(missing)}/{len(results)} videos; missing: {missing}")


if __name__ == "__main__":
    main()
