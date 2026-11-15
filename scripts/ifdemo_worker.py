"""IF demo worker (runs ON a Lambda GPU): paired legacy vs corrected-controller episodes, one GPU, seed 42.

    IFD_CASES=IF007,IF043 CUDA_VISIBLE_DEVICES=0 python scripts/ifdemo_worker.py

Legacy = frozen full-run config (`5b8a5b8`, pushed to ~/legacy): lingbot_rollout --text-prompt both, 40 chunks,
--player-retries 2. Corrected = current main's lingbot_released_baseline --generate --validate-controller, 60 chunks.
Each episode runs in a fresh process; one status file per attempt; an existing status is never overwritten.
Mirrors scripts/run_codex_ccv.py (kept separate because live Codex workers are hash-pinned to that file).
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LEGACY = Path.home() / "legacy"


def main():
    output = ROOT / "outputs/ifdemo"
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", WORLDSYNC_HOLD_MAX_TOKENS="400")
    tag = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
    for task in [t for t in os.environ["IFD_CASES"].split(",") if t]:
        for arm in ("L", "C"):
            episode = output / arm / f"if-{task}"
            status = output / f"{arm}-if-{task}-status.json"
            if status.exists() and json.loads(status.read_text()).get("ok"):
                continue
            if arm == "C":  # adopt an earlier complete run (an older check looked for the video in the wrong folder)
                done = sorted(d for d in output.joinpath(arm).glob(f"if-{task}-failed*")
                              if (d / "episode/rollout.mp4").exists() and json.loads((d / "episode/run.json").read_text()).get("status") == "completed")
                if done:
                    if episode.exists():
                        episode.rename(episode.with_name(episode.name + f"-dup{int(time.time())}"))
                    done[0].rename(episode)
                    status.write_text(json.dumps({"arm": arm, "task_id": task, "ok": True, "adopted": done[0].name}, indent=2) + "\n")
                    continue
            if arm == "C":
                cwd, pythonpath = ROOT, ROOT
                args = ["-m", "worldsync.lingbot_released_baseline", "--split", "if", "--task-id", task, "--seed", "42",
                        "--chunks", "60", "--generate", "--validate-controller", "--output", str(episode)]
            else:
                cwd, pythonpath = LEGACY, LEGACY
                args = ["-m", "worldsync.lingbot_rollout", "--split", "if", "--task-id", task, "--seed", "42",
                        "--max-chunks", "40", "--text-prompt", "both", "--player-retries", "2", "--offload-text-encoder",
                        "--output", str(episode)]
            for attempt in (1, 2):
                if episode.exists():
                    episode.rename(episode.with_name(episode.name + f"-failed{int(time.time())}"))
                started = time.time()
                with (output / f"{arm}-if-{task}.log").open("a") as log:
                    result = subprocess.run([sys.executable, *args], cwd=cwd, env=dict(env, PYTHONPATH=str(pythonpath)),
                                            stdout=log, stderr=subprocess.STDOUT)
                video = episode / "episode" / "rollout.mp4" if arm == "C" else episode / "rollout.mp4"
                ok = result.returncode == 0 and video.exists()
                if ok:
                    break
                time.sleep(30)
            record = {"arm": arm, "task_id": task, "returncode": result.returncode, "ok": ok, "attempts": attempt,
                      "gpu": tag, "elapsed_seconds": round(time.time() - started, 1), "argv": args}
            status.write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps(record), flush=True)
    (output / f"worker-{tag}-DONE").write_text("done\n")


if __name__ == "__main__":
    main()
