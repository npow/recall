"""Demo: three agents in one LingBot-World2 world sharing RECALL pose retrieval.

The viewer runs first and exports its archive; each dueler retrieves from its own
archive and the viewer's. One container keeps the ~20 GB archive on local disk.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if modal.is_local():
    from scripts.modal_lingbot_benchmark import image, weights, results
else:
    image = None
    weights = modal.Volume.from_name("worldsync-lingbot-weights")
    results = modal.Volume.from_name("worldsync-lingbot-results")

app = modal.App("worldsync-lingbot-trio")
TEXT = ("Two cowboy gunslingers stand in the dusty street facing each other, "
        "slowly backing away from each other, a western duel standoff.")
AGENTS = [("viewer", "trio_viewer", []), ("duel-a", "trio_duel_a", ["viewer"]), ("duel-b", "trio_duel_b", ["viewer"])]


@app.function(image=image, gpu="H200", cpu=8, memory=131072, timeout=3600,
              volumes={"/cache": weights, "/results": results})
def trio(seed: int = 42, task_id: str = "GC010", text: str = TEXT, turn_rate: float = 60, only: str = ""):
    from worldsync.lingbot_paths import PATHS
    weights.reload()
    target = Path("/work/external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast")
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        target.symlink_to("/cache/14b-causal-fast")
    archives = Path("/tmp/archives")
    archives.mkdir(exist_ok=True)
    batch = f"trio-{task_id}-seed{seed}-{time.time_ns()}"
    done = {}
    for name, path, sources in AGENTS:
        if only and name not in only.split(","):
            continue
        actions = PATHS[path]
        output = Path("/work/outputs") / batch / name
        command = [sys.executable, "-u", "-m", "worldsync.lingbot_rollout", "--split", "gc",
                   "--task-id", task_id, "--seed", str(seed), "--turn-degrees-per-second", str(turn_rate),
                   "--max-chunks", str(len(actions)), "--native-fixed-controls", "--fixed-actions", *actions,
                   "--pose-retrieval", "--world-text-suffix", text, "--output", str(output)]
        if name == "viewer":
            command += ["--archive-export", str(archives / "viewer.pt")]
        for source in sources:
            command += ["--foreign-archive", f"{source}={archives / (source + '.pt')}"]
        outcome = subprocess.run(command, cwd="/work")
        if output.exists():
            shutil.copytree(output, Path("/results") / batch / name)
            results.commit()
        print(subprocess.run(["df", "-h", "/tmp"], capture_output=True, text=True).stdout, flush=True)
        if outcome.returncode:
            raise RuntimeError(f"{name} failed; artifacts preserved: {batch}/{name}")
        done[name] = json.loads((output / "run.json").read_text())["status"]
    return {"batch": batch, "agents": done}


@app.local_entrypoint()
def main(seed: int = 42, task_id: str = "GC010", only: str = ""):
    print(json.dumps(trio.remote(seed=seed, task_id=task_id, only=only)))
