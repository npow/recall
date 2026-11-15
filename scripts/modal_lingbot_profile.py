"""Collect a warmed CPU/CUDA trace on one H200 using the verified cached weights."""
import json
from pathlib import Path
import sys
import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.modal_lingbot_benchmark import image, weights, results

app = modal.App("worldsync-lingbot-profile")


@app.function(image=image, gpu="H200", cpu=8, memory=131072, timeout=1800,
              max_containers=1, volumes={"/cache": weights, "/results": results})
def collect():
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
        outcome = subprocess.run(command, cwd="/work")
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
def main():
    report = collect.remote()
    output = ROOT / "outputs/lingbot-reproduction/modal-h200-profile.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "operators"}, indent=2))
