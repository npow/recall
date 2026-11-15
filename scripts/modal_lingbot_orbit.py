"""Demo: two first-person agents circling each other in one LingBot-World2 world.

Agent A orbits first and exports its full-lap archive; agent B orbits half a lap
behind on the same circle and retrieves A's views at its own poses.
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

app = modal.App("worldsync-lingbot-orbit")
TEXT = ("an opponent wrestler stands a few meters ahead facing you on the sand of a Roman coliseum arena, "
        "the two of you slowly circling each other")
CAPTION = ("First-person view on the sand floor of an ancient Roman coliseum arena: a gladiator in leather armor "
           "stands a few meters ahead facing the camera in a wrestling stance, with curved stone walls, red banners "
           "and tiered seating with spectators behind him under a clear blue sky.")
AGENTS = [("agent-a", "arena", [], None), ("agent-b", "arena", ["agent-a"], ["0", "8", "180"])]
EXPORTERS = {"agent-a"}


@app.function(image=image, gpu="H200", cpu=8, memory=131072, timeout=3600,
              volumes={"/cache": weights, "/results": results})
def orbit(image_bytes: bytes, image_b_bytes: bytes = b"", seed: int = 42, task_id: str = "GC010", text: str = TEXT, turn_rate: float = 60,
          only: str = "", archive_until: int = 40, archive_stride: int = 1, min_retrieved: int = 1):
    from worldsync.lingbot_paths import PATHS
    weights.reload()
    target = Path("/work/external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast")
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        target.symlink_to("/cache/14b-causal-fast")
    archives = Path("/tmp/archives")
    archives.mkdir(exist_ok=True)
    batch = f"orbit-{task_id}-seed{seed}-{time.time_ns()}"
    data_root = Path("/tmp/cleandata")
    (data_root / "gc/clean").mkdir(parents=True, exist_ok=True)
    tasks = json.loads(Path("/work/external/PlayWorld/data/gc/data.json").read_text())
    for record in tasks:
        if record["task_id"] == task_id:
            record["image_path"] = f"gc/clean/{task_id}.jpg"
            record["image_caption"] = CAPTION
    (data_root / "gc/data.json").write_text(json.dumps(tasks))
    (data_root / f"gc/clean/{task_id}.jpg").write_bytes(image_bytes)
    # Agent B starts from A's halfway frame (opposite side of the circle, facing A), via its own data root.
    data_root_b = Path("/tmp/cleandata-b")
    (data_root_b / "gc/clean").mkdir(parents=True, exist_ok=True)
    (data_root_b / "gc/data.json").write_text(json.dumps(tasks))
    (data_root_b / f"gc/clean/{task_id}.jpg").write_bytes(image_b_bytes or image_bytes)
    done = {}
    for name, path, sources, start_pose in AGENTS:
        if only and name not in only.split(","):
            continue
        actions = PATHS[path]
        output = Path("/work/outputs") / batch / name
        command = [sys.executable, "-u", "-m", "worldsync.lingbot_rollout", "--split", "gc",
                   "--task-id", task_id, "--seed", str(seed), "--turn-degrees-per-second", str(turn_rate),
                   "--max-chunks", str(len(actions)), "--native-fixed-controls", "--fixed-actions", *actions,
                   "--pose-retrieval", "--world-text-suffix", text, "--output", str(output),
                   "--data-root", str(data_root_b if name == "agent-b" else data_root), "--archive-until", str(archive_until),
                   "--archive-stride", str(archive_stride), "--min-retrieved-frames", str(min_retrieved)]
        if start_pose:
            command += ["--start-pose", *start_pose]
        if name in EXPORTERS:
            command += ["--archive-export", str(archives / f"{name}.pt")]
        print(json.dumps({"agent": name, "command": command}), flush=True)
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
def main(image: str, image_b: str = "", seed: int = 42, task_id: str = "GC010", only: str = ""):
    b = Path(image_b).read_bytes() if image_b else b""
    print(json.dumps(orbit.remote(Path(image).read_bytes(), b, seed=seed, task_id=task_id, only=only)))
