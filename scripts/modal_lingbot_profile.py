"""Collect a warmed CPU/CUDA trace on one H200 using the verified cached weights."""
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

app = modal.App("worldsync-lingbot-profile")


@app.function(image=image, gpu="H200", cpu=8, memory=131072, timeout=1800,
              max_containers=1, volumes={"/cache": weights, "/results": results})
def collect(fast_metadata: bool = False):
    import os
    import gzip
    import shutil
    import subprocess
    import time
    weights.reload()
    target = Path("/work/external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast")
    target.parent.mkdir(exist_ok=True)
    target.symlink_to("/cache/14b-causal-fast")
    identifier = f"profile-h200-{time.time_ns()}"
    output = Path("/work/outputs") / identifier
    command = [sys.executable, "-u", "-m", "worldsync.lingbot_profile", "--split", "gc", "--task-id",
               "GC007", "--max-chunks", "10", "--fixed-actions", *(["WAIT"] * 10), "--output", str(output)]
    try:
        environment = dict(os.environ, WORLDSYNC_HOST_CACHE_INDICES="1" if fast_metadata else "0")
        outcome = subprocess.run(command, cwd="/work", env=environment)
    finally:
        if output.exists():
            trace = output / "trace.json"
            if trace.exists():
                with trace.open("rb") as source, gzip.open(output / "trace.json.gz", "wb") as destination:
                    shutil.copyfileobj(source, destination)
                trace.unlink()
            shutil.copytree(output, Path("/results") / identifier)
            results.commit()
    if outcome.returncode:
        raise RuntimeError(f"Profiler failed; artifacts retained at {identifier}")
    report = json.loads((output / "profile.json").read_text())
    report["result_directory"] = identifier
    run = json.loads((output / "run.json").read_text())
    report["chunks"] = run["chunks"]
    return report


@app.local_entrypoint()
def main(fast_metadata: bool = False):
    report = collect.remote(fast_metadata)
    filename = "modal-h200-profile-fast-metadata.json" if fast_metadata else "modal-h200-profile.json"
    output = ROOT / "outputs/lingbot-reproduction" / filename
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "operators"}, indent=2))
