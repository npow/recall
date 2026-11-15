"""One complete GC007 stock episode, followed by the local Claude rubric."""
import json
from pathlib import Path
import sys
import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if modal.is_local():
    import os
    from scripts.modal_lingbot_benchmark import runtime_image, weights, results
    from worldsync.credentials import load_local_credentials
    load_local_credentials()
    credentials = modal.Secret.from_dict({"ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"]})
    image = (runtime_image.pip_install("anthropic==2.8.0")
        .run_commands("git clone https://github.com/hku-sail/PlayWorld.git /work/external/PlayWorld",
                      "git -C /work/external/PlayWorld checkout eeaea12e050a90b815d11d030e943e7b3181da21")
        .add_local_dir(ROOT / "worldsync", "/work/worldsync", ignore=["__pycache__"])
        .add_local_file(ROOT / "reproduction/lingbot-weights.json", "/work/reproduction/lingbot-weights.json")
        .add_local_file(ROOT / "external/PlayWorld/data/gc/data.json", "/work/external/PlayWorld/data/gc/data.json")
        .add_local_file(ROOT / "external/PlayWorld/data/gc/images/GC007.jpg", "/work/external/PlayWorld/data/gc/images/GC007.jpg"))
else:
    image = None
    credentials = modal.Secret.from_dict({})
    weights = modal.Volume.from_name("worldsync-lingbot-weights")
    results = modal.Volume.from_name("worldsync-lingbot-results")

app = modal.App("worldsync-lingbot-single-episode")


@app.function(image=image, gpu="H200", cpu=8, memory=131072, timeout=1800, max_containers=1,
              secrets=[credentials], volumes={"/cache": weights, "/results": results})
def generate():
    import shutil
    import subprocess
    import time
    weights.reload()
    target = Path("/work/external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast")
    target.parent.mkdir(exist_ok=True)
    target.symlink_to("/cache/14b-causal-fast")
    identifier = f"gc-GC007-v3-{time.time_ns()}"
    output = Path("/work/outputs") / identifier
    try:
        outcome = subprocess.run([sys.executable, "-u", "-m", "worldsync.lingbot_rollout", "--split", "gc",
            "--task-id", "GC007", "--max-chunks", "40", "--output", str(output)], cwd="/work")
    finally:
        if output.exists():
            shutil.copytree(output, Path("/results") / identifier)
            results.commit()
    if outcome.returncode:
        raise RuntimeError(f"Episode generation failed; preserved artifacts: {identifier}")
    return identifier


@app.function(image=image, cpu=4, memory=8192, timeout=900, secrets=[credentials], volumes={"/results": results})
def score(identifier: str):
    import subprocess
    results.reload()
    output = Path("/results") / identifier
    data = Path("/work/external/PlayWorld/data")
    task = next(t for t in json.loads((data / "gc/data.json").read_text()) if t["task_id"] == "GC007")
    outcome = subprocess.run([sys.executable, "-u", "-m", "worldsync.judge", "--dataset", str(data / "gc/data.json"),
        "--task-id", "GC007", "--video", str(output / "rollout.mp4"), "--reference-image", str(data / task["image_path"]),
        "--world-model", "LingBot-World2 14B causal-fast stock", "--output", str(output / "judge.json")], cwd="/work")
    results.commit()
    if outcome.returncode:
        raise RuntimeError(f"Episode scoring failed; preserved artifacts: {identifier}")
    run = json.loads((output / "run.json").read_text())
    judge = json.loads((output / "judge.json").read_text())
    return {"scope": "single complete stock episode; local Claude protocol, not leaderboard reproduction",
        "result_directory": identifier, "protocol_id": run["protocol_id"], "task_id": "GC007",
        "generated_seconds": run["generated_seconds"], "termination": run["termination"],
        "elapsed_s": run["elapsed_s"], "actions": run["actions"], "judge": judge}


@app.local_entrypoint()
def main():
    identifier = generate.remote()
    print("Complete generation saved:", identifier, flush=True)
    report = score.remote(identifier)
    destination = ROOT / "outputs/lingbot-reproduction/modal-single-episode.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"task": report["task_id"], "score": report["judge"]["reported_score"],
                      "gate": report["judge"]["gate"], "result_directory": identifier}, indent=2))
