"""Stock native-control round-trip calibration; no adaptive player or memory edit."""
import json
from pathlib import Path
import sys
import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if modal.is_local():
    from scripts.modal_lingbot_benchmark import image, weights, results
else:
    image = None
    weights = modal.Volume.from_name("worldsync-lingbot-weights")
    results = modal.Volume.from_name("worldsync-lingbot-results")

app = modal.App("worldsync-lingbot-roundtrip")


@app.function(image=image, gpu="H200", cpu=8, memory=131072, timeout=1800, max_containers=1,
              volumes={"/cache": weights, "/results": results})
def generate(path_name: str = "round_trip", seed: int = 42, retrieval: bool = False):
    import shutil
    import subprocess
    import time
    from worldsync.lingbot_paths import PATHS
    actions = PATHS[path_name]
    weights.reload()
    target = Path("/work/external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast")
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        target.symlink_to("/cache/14b-causal-fast")
    mode = "retrieval" if retrieval else "stock"
    identifier = f"{mode}-{path_name}-seed{seed}-{time.time_ns()}"
    output = Path("/work/outputs") / identifier
    try:
        command = [sys.executable, "-u", "-m", "worldsync.lingbot_rollout", "--split", "gc",
            "--task-id", "GC007", "--seed", str(seed), "--max-chunks", str(len(actions)),
            "--native-fixed-controls", "--fixed-actions", *actions, "--output", str(output)]
        if retrieval:
            command.append("--pose-retrieval")
        outcome = subprocess.run(command, cwd="/work")
    finally:
        if output.exists():
            shutil.copytree(output, Path("/results") / identifier)
            results.commit()
    if outcome.returncode:
        raise RuntimeError(f"Calibration failed; artifacts preserved: {identifier}")
    report = json.loads((output / "run.json").read_text())
    return {"scope": "fixed-path native stock calibration; not PlayWorld task score",
            "path": path_name, "result_directory": identifier, "run": report}


@app.local_entrypoint()
def main(path_name: str = "round_trip", seed: int = 42, retrieval: bool = False):
    result = generate.remote(path_name, seed, retrieval)
    mode = "retrieval" if retrieval else "stock"
    output = ROOT / "outputs/lingbot-reproduction" / f"native-{mode}-{path_name}-seed{seed}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "run"}, indent=2))
