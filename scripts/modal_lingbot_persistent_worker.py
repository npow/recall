"""Single-worker checkpoint reuse pilot in the verified secondary workspace."""
import hashlib
import json
from pathlib import Path
import sys

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if modal.is_local():
    from scripts.modal_lingbot_benchmark import image as runtime_image, weights, results
    image = runtime_image.add_local_file(ROOT / "reproduction/lingbot-runtime-benchmark/sdpa26.json",
                                         "/work/reproduction/fresh-sdpa26.json")
else:
    image = None
    weights = modal.Volume.from_name("worldsync-lingbot-weights")
    results = modal.Volume.from_name("worldsync-lingbot-results")

app = modal.App("worldsync-lingbot-persistent-worker")


@app.function(image=image, gpu="H100", cpu=8, memory=131072, timeout=1800, max_containers=1,
              volumes={"/cache": weights, "/results": results})
def probe(workspace: str, verification: dict | None = None):
    import os
    import shutil
    import subprocess
    import time
    manifest = json.loads(Path("/work/reproduction/lingbot-weights.json").read_text())
    if workspace not in {"nissan-pow", "krasnaya-zvesda"}:
        raise ValueError("Undeclared workspace")
    if workspace != "nissan-pow":
        expected = [{key: row[key] for key in ("path", "bytes", "sha256")} for row in manifest["files"]]
        if (verification is None or verification["status"] != "verified" or verification["workspace"] != workspace
                or verification["revision"] != manifest["revision"] or verification["files"] != expected):
            raise ValueError("Secondary checkpoint must be CPU SHA-verified before GPU execution")
    weights.reload()
    raw = Path("/cache/14b-causal-fast")
    if any((raw / row["path"]).stat().st_size != row["bytes"] for row in manifest["files"]):
        raise ValueError("Verified staging no longer matches checkpoint inventory")
    target = Path("/work/external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast")
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        target.symlink_to(raw)
    identifier = "persistent-worker-" + str(time.time_ns())
    output = Path("/work/outputs") / identifier
    try:
        outcome = subprocess.run([sys.executable, "-u", "-m", "worldsync.lingbot_persistent_probe",
                                  "--output", str(output)], cwd="/work",
                                 env=dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                                          WORLDSYNC_LINGBOT_ATTENTION="sdpa"))
    finally:
        if output.exists():
            shutil.copytree(output, Path("/results") / identifier)
            results.commit()
    return {"status": "completed" if outcome.returncode == 0 else "failed",
            "workspace": workspace, "result_directory": identifier,
            "verification_record_sha256": (hashlib.sha256(json.dumps(verification, sort_keys=True).encode()).hexdigest()
                                             if verification else None),
            "verification_scope": ("new CPU SHA verification" if verification else "reuse previously verified primary baseline checkpoint; byte-size inventory rechecked"),
            "summary": json.loads((output / "summary.json").read_text())}


@app.local_entrypoint()
def main(workspace: str = "nissan-pow"):
    verification = None
    if workspace != "nissan-pow":
        verification = json.loads((ROOT / f"outputs/lingbot-reproduction/weights-verified-{workspace}.json").read_text())
        if verification["status"] != "verified" or verification["workspace"] != workspace:
            raise ValueError("Do not allocate a GPU before secondary checkpoint verification")
    report = probe.remote(workspace, verification)
    output = ROOT / "reproduction/lingbot-persistent-worker"
    output.mkdir(exist_ok=True)
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "completed":
        raise RuntimeError("Persistent worker parity failed; complete attempt preserved")
