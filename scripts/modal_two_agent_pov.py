"""Two first-person LingBot histories, shared handoff intent, labeled tiled MP4.

modal run --profile nissan-pow scripts/modal_two_agent_pov.py
One ephemeral H200, two serial 12-chunk renders, one warm model, 20-minute cap.
"""
import json
from pathlib import Path
import sys
import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if modal.is_local():
    from scripts.modal_lingbot_benchmark import runtime_image, weights, results
    from worldsync.pov_video import prepare
    prepare(ROOT / "outputs/two-agent-pov/inputs")
    image = (runtime_image
             .add_local_dir(ROOT / "worldsync", "/work/worldsync", ignore=["__pycache__"])
             .add_local_file(ROOT / "reproduction/lingbot-weights.json", "/work/reproduction/lingbot-weights.json")
             .add_local_file(ROOT / "reproduction/state.json", "/work/reproduction/state.json")
             .add_local_dir(ROOT / "outputs/two-agent-pov/inputs", "/work/pov-inputs"))
else:
    image = None
    weights = modal.Volume.from_name("worldsync-lingbot-weights")
    results = modal.Volume.from_name("worldsync-lingbot-results")

app = modal.App("worldsync-two-agent-pov")


@app.function(image=image, gpu="H200", cpu=8, memory=131072, timeout=1200,
              max_containers=1, volumes={"/cache": weights, "/results": results})
def generate():
    import os
    import shutil
    import time
    from worldsync.lingbot_runtime import WarmRuntime
    from worldsync.lingbot_rollout import main as rollout
    from worldsync.pov_video import ACTORS, CHUNKS
    os.environ["WORLDSYNC_LINGBOT_ATTENTION"] = "sdpa"
    weights.reload()
    raw = Path("/cache/14b-causal-fast")
    manifest = json.loads(Path("/work/reproduction/lingbot-weights.json").read_text())
    for row in manifest["files"]:
        if (raw / row["path"]).stat().st_size != row["bytes"]:
            raise ValueError("checkpoint inventory mismatch")
    target = Path("/work/external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to(raw)
    identifier = f"two-agent-pov-{time.time_ns()}"
    root = Path("/work/outputs") / identifier
    runtime = WarmRuntime()
    outcomes = {}
    for actor, seed in zip(ACTORS, (42, 43)):
        output = root / actor
        try:
            run = rollout(["--data-root", f"/work/pov-inputs/{actor}", "--split", "insight", "--task-id", "POV001",
                           "--max-chunks", str(CHUNKS), "--seed", str(seed), "--fixed-actions", *(["WAIT"]*CHUNKS),
                           "--director", "perchunk", "--director-text-file", f"/work/pov-inputs/{actor}-director.json",
                           "--offload-text-encoder", "--output", str(output)], runtime=runtime)
            if run["status"] != "completed" or len(run["chunks"]) != CHUNKS:
                raise ValueError(f"{actor} did not finish all chunks")
            outcomes[actor] = {"run": run, "video": (output / "rollout.mp4").read_bytes()}
        finally:
            if output.exists():
                shutil.copytree(output, Path("/results") / identifier / actor)
                results.commit()
            runtime.between_jobs()
    return {"directory": identifier, "views": outcomes}


@app.local_entrypoint()
def main(output: str = "outputs/two-agent-pov"):
    from worldsync.pov_video import tile
    result = generate.remote()
    root = ROOT / output
    root.mkdir(parents=True, exist_ok=True)
    for actor, outcome in result.pop("views").items():
        (root / f"{actor}.mp4").write_bytes(outcome["video"])
        (root / f"{actor}-run.json").write_text(json.dumps(outcome["run"], indent=2)+"\n")
    composition = tile(root / "orange.mp4", root / "purple.mp4", root / "tiled.mp4")
    (root / "result.json").write_text(json.dumps({**result, **composition}, indent=2)+"\n")
    print(json.dumps({"directory": result["directory"], "video": str(root / "tiled.mp4"),
                      "seconds": composition["duration"]}, indent=2))
