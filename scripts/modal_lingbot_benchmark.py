"""One H200 timing pilot. Run with --profile nissan-pow; no agent API calls."""
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "1895d300d8ac936401689b26389f51cbd36530eb"
WEIGHT_REVISION = "5c33dd40b213598c418fd25bff30fdbd23fd38a7"
app = modal.App("worldsync-lingbot-timing")
weights = modal.Volume.from_name("worldsync-lingbot-weights", create_if_missing=True)
results = modal.Volume.from_name("worldsync-lingbot-results", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.6.0", "torchvision==0.21.0",
                 index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("transformers==4.51.3", "tokenizers==0.21.4", "numpy==1.26.4",
                 "diffusers==0.37.0", "huggingface_hub==0.36.2", "hf_xet",
                 "accelerate", "safetensors", "Pillow", "einops", "easydict", "ftfy",
                 "sentencepiece", "tqdm", "imageio", "imageio-ffmpeg", "opencv-python-headless==4.11.0.86")
    .run_commands("mkdir -p /work/external/LingBot-World2",
                  "git -C /work/external/LingBot-World2 init",
                  "git -C /work/external/LingBot-World2 remote add origin https://github.com/Robbyant/lingbot-world-v2.git",
                  f"git -C /work/external/LingBot-World2 fetch --depth 1 origin {SOURCE_REVISION}",
                  "git -C /work/external/LingBot-World2 checkout FETCH_HEAD")
    .env({"PYTHONPATH": "/work", "HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_DOWNLOAD_TIMEOUT": "180"})
    .add_local_dir(ROOT / "worldsync", "/work/worldsync", ignore=["__pycache__"])
    .add_local_file(ROOT / "reproduction/lingbot-weights.json", "/work/reproduction/lingbot-weights.json")
    .add_local_file(ROOT / "external/PlayWorld/data/gc/data.json", "/work/external/PlayWorld/data/gc/data.json")
    .add_local_file(ROOT / "external/PlayWorld/data/gc/images/GC007.jpg", "/work/external/PlayWorld/data/gc/images/GC007.jpg")
)


@app.function(image=image, volumes={"/cache": weights}, cpu=8, memory=16384, timeout=3600)
def prepare():
    """Download in the cloud before allocating a GPU; verify pinned file hashes."""
    import hashlib
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import hf_hub_download, configure_http_backend
    import requests
    def backend():
        session = requests.Session()
        session.headers["Accept-Encoding"] = "identity"
        return session
    configure_http_backend(backend_factory=backend)
    manifest = json.loads(Path("/work/reproduction/lingbot-weights.json").read_text())
    target = Path("/cache/14b-causal-fast")
    target.mkdir(exist_ok=True)
    started = time.monotonic()
    def fetch(item):
        path = hf_hub_download("robbyant/lingbot-world-v2-14b-causal-fast", item["path"],
            revision=WEIGHT_REVISION, local_dir=target)
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
        if os.path.getsize(path) != item["bytes"] or digest.hexdigest() != item["sha256"]:
            raise ValueError(f"Pinned weight verification failed: {item['path']}")
        print("Verified", item["path"], flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, manifest["files"]))
    weights.commit()
    return {"download_and_verification_s": time.monotonic() - started, "files": len(manifest["files"])}


@app.function(image=image, gpu="H200", cpu=8, memory=131072, timeout=1800,
              max_containers=1, volumes={"/cache": weights, "/results": results})
def benchmark():
    import os
    import subprocess
    import sys
    import time
    import torch
    weights.reload()
    target = Path("/work/external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast")
    target.parent.mkdir(exist_ok=True)
    target.symlink_to("/cache/14b-causal-fast")
    identifier = f"h200-{time.time_ns()}"
    output = Path("/work/outputs") / identifier
    started = time.monotonic()
    command = [sys.executable, "-u", "-m", "worldsync.lingbot_rollout", "--split", "gc",
               "--task-id", "GC007", "--max-chunks", "8", "--fixed-actions",
               "WAIT", "W", "RIGHT", "WAIT", "W", "LEFT", "WAIT", "W", "--output", str(output)]
    outcome = subprocess.run(command, cwd="/work")
    report = json.loads((output / "run.json").read_text()) if (output / "run.json").exists() else {}
    chunks = report.get("chunks", [])
    intervals = [b["elapsed_s"] - a["elapsed_s"] for a, b in zip(chunks, chunks[1:])]
    summary = {"scope": "fixed-action hardware timing; no baseline score or adaptive player",
        "status": report.get("status", "failed"), "returncode": outcome.returncode,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "wall_s": time.monotonic() - started, "generated_seconds": report.get("generated_seconds"),
        "chunk_intervals_s": intervals, "peak_allocated_bytes": report.get("peak_allocated_bytes"),
        "source_revision": SOURCE_REVISION, "checkpoint_revision": WEIGHT_REVISION,
        "deviations": ["Torch 2.6.0/CUDA 12.4 on H200 vs local Torch 2.15.0/CUDA 13.2 on Thor"],
        "result_volume": "worldsync-lingbot-results", "result_directory": identifier}
    (output / "timing.json").write_text(json.dumps(summary, indent=2) + "\n")
    import shutil
    shutil.copytree(output, Path("/results") / identifier)
    results.commit()
    if outcome.returncode:
        raise RuntimeError(f"Timing pilot failed; artifacts retained in {identifier}")
    return summary


@app.local_entrypoint()
def main():
    preparation = prepare.remote()
    result = benchmark.remote()
    result["preparation"] = preparation
    destination = ROOT / "outputs/lingbot-reproduction/modal-h200-timing.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
