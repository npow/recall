"""Generate an actual LingBot MP4 from two character policies' shared-state plan.

modal run --profile nissan-pow scripts/modal_multi_agent_video.py
One H200, one 24-chunk rollout, 20-minute function timeout, no deployed service.
"""
import json
from pathlib import Path
import sys
import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if modal.is_local():
    from scripts.modal_lingbot_benchmark import runtime_image, weights, results
    image = (runtime_image
             .add_local_dir(ROOT / "worldsync", "/work/worldsync", ignore=["__pycache__"])
             .add_local_file(ROOT / "reproduction/lingbot-weights.json", "/work/reproduction/lingbot-weights.json")
             .add_local_file(ROOT / "reproduction/state.json", "/work/reproduction/state.json")
             .add_local_file(ROOT / "external/PlayWorld/data/insight/data.json", "/work/external/PlayWorld/data/insight/data.json")
             .add_local_file(ROOT / "external/PlayWorld/data/insight/images/OE025.jpg", "/work/external/PlayWorld/data/insight/images/OE025.jpg"))
else:
    image = None
    weights = modal.Volume.from_name("worldsync-lingbot-weights")
    results = modal.Volume.from_name("worldsync-lingbot-results")

app = modal.App("worldsync-multi-agent-video")


@app.function(image=image, gpu="H200", cpu=8, memory=131072, timeout=1200,
              max_containers=1, volumes={"/cache": weights, "/results": results})
def generate(director: dict, agent_plan: dict):
    import os
    import shutil
    import subprocess
    import time
    weights.reload()
    raw = Path("/cache/14b-causal-fast")
    manifest = json.loads(Path("/work/reproduction/lingbot-weights.json").read_text())
    for row in manifest["files"]:
        if (raw / row["path"]).stat().st_size != row["bytes"]:
            raise ValueError("Checkpoint inventory mismatch")
    target = Path("/work/external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to(raw)
    identifier = f"multi-agent-video-{time.time_ns()}"
    output = Path("/work/outputs") / identifier
    texts = Path("/work/multi-agent-director.json")
    texts.write_text(json.dumps(director))
    command = [sys.executable, "-u", "-m", "worldsync.lingbot_rollout", "--split", "insight",
               "--task-id", "OE025", "--max-chunks", "24", "--seed", "42",
               "--fixed-actions", *(["WAIT"]*24), "--director", "perchunk",
               "--director-text-file", str(texts), "--offload-text-encoder", "--output", str(output)]
    try:
        result = subprocess.run(command, cwd="/work", env=dict(os.environ, WORLDSYNC_LINGBOT_ATTENTION="sdpa"))
    finally:
        if output.exists():
            (output / "agent-plan.json").write_text(json.dumps(agent_plan, indent=2)+"\n")
            (output / "director-input.json").write_text(json.dumps(director, indent=2)+"\n")
            shutil.copytree(output, Path("/results") / identifier)
            results.commit()
    if result.returncode:
        raise RuntimeError(f"Generation failed; preserved artifacts: {identifier}")
    run = json.loads((output / "run.json").read_text())
    return {"directory": identifier, "status": run["status"], "seconds": run["generated_seconds"],
            "chunks": len(run["chunks"]), "video": (output / "rollout.mp4").read_bytes(),
            "run": run}


@app.local_entrypoint()
def main(plan_dir: str = "outputs/multi-agent-video-plan", output: str = "outputs/multi-agent-world-video"):
    plan = ROOT / plan_dir
    result = generate.remote(json.loads((plan / "director.json").read_text()),
                             json.loads((plan / "agent-plan.json").read_text()))
    destination = ROOT / output
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "rollout.mp4").write_bytes(result.pop("video"))
    (destination / "run.json").write_text(json.dumps(result.pop("run"), indent=2)+"\n")
    (destination / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    for name in ("agent-plan.json", "director.json"):
        (destination / name).write_bytes((plan / name).read_bytes())
    print(json.dumps(result, indent=2))
