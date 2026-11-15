"""Run the full run's STOCK arm at seed 43 on the local Thor GPU, one episode at a time.

Same frozen flags as reproduction/full-run-config.json (stock arm): --player-retries 2 --text-prompt both
--offload-text-encoder, max 40 chunks (60 for insight); insight uses --text-prompt caption per the
pre-declared insight amendment. Only the seed (43) and the hardware (Thor, not Modal) differ.

Resumable: an episode with outputs/thor-seed43/episodes/<id>/rollout.mp4 and run.json is skipped.
Pauses (never competes) while another local GPU job (matrix or lingbot rollout) is running.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/npow/code/iteratehack/.venv-lingbot/bin/python"
SEED = 43
MAX_CHUNKS = {"gc": 40, "if": 40, "outsight": 40, "insight": 60}
OUT = ROOT / "outputs/thor-seed43"
ORDER = ("if", "gc", "outsight", "insight")  # cheapest first; 60-chunk insight last


def other_gpu_jobs():
    """Local GPU jobs that are not ours (matrix suite or another lingbot rollout)."""
    listing = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()
    busy = []
    for line in listing:
        if ("worldsync.lingbot_rollout" in line or "worldsync.matrix" in line or "lingbot_native" in line) \
                and "thor-seed43" not in line and "--dry-run" not in line and "ps -eo" not in line:
            busy.append(line.strip()[:120])
    return busy


def episode_args(split, task):
    prompt = "caption" if split == "insight" else "both"
    return ["--split", split, "--task-id", task, "--seed", str(SEED), "--max-chunks", str(MAX_CHUNKS[split]),
            "--player-retries", "2", "--text-prompt", prompt, "--offload-text-encoder"]


def main():
    cases = json.loads((ROOT / "outputs/thor-cases.json").read_text())
    cases.sort(key=lambda c: ORDER.index(c["split"]))
    (OUT / "episodes").mkdir(parents=True, exist_ok=True)
    for case in cases:
        split, task = case["split"], case["task_id"]
        name = f"stock-{split}-{task}-seed{SEED}"
        directory = OUT / "episodes" / name
        if (directory / "rollout.mp4").is_file() and (directory / "run.json").is_file():
            print(json.dumps({"event": "skip", "episode": name}), flush=True)
            continue
        while other_gpu_jobs():
            print(json.dumps({"event": "paused_other_gpu_job", "jobs": other_gpu_jobs()[:2]}), flush=True)
            time.sleep(120)
        if directory.exists():  # partial earlier attempt: keep the log, restart the episode cleanly
            stale = directory.with_name(name + f".partial-{int(time.time())}")
            directory.rename(stale)
        started = time.time()
        print(json.dumps({"event": "start", "episode": name}), flush=True)
        with open(OUT / "episodes" / f"{name}.log", "ab") as log:
            result = subprocess.run([PYTHON, "-u", "-m", "worldsync.lingbot_rollout", *episode_args(split, task),
                                     "--output", str(directory)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps({"event": "done", "episode": name, "returncode": result.returncode,
                          "seconds": round(time.time() - started)}), flush=True)
    print(json.dumps({"event": "all_done"}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
