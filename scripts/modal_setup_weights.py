"""Populate a Modal volume with the pinned official LingBot 14B causal-fast weights.

    MODAL_PROFILE=<workspace> python -m modal run scripts/modal_setup_weights.py

Downloads inside Modal (datacenter bandwidth, no local disk), then checks every
file size against reproduction/lingbot-weights.json. CPU-only; no GPU spend.
"""
import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REPO = "robbyant/lingbot-world-v2-14b-causal-fast"
REVISION = "5c33dd40b213598c418fd25bff30fdbd23fd38a7"
app = modal.App("worldsync-lingbot-setup")
weights = modal.Volume.from_name("worldsync-lingbot-weights", create_if_missing=True)
results = modal.Volume.from_name("worldsync-lingbot-results", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub", "hf_xet")
    .add_local_file(ROOT / "reproduction/lingbot-weights.json", "/root/lingbot-weights.json"))


@app.function(image=image, volumes={"/weights": weights}, timeout=3 * 3600, cpu=8, memory=16384,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})])
def download():
    from huggingface_hub import snapshot_download
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    target = "/weights/14b-causal-fast"
    snapshot_download(repo_id=REPO, revision=REVISION, local_dir=target, max_workers=16,
        allow_patterns=["transformers/*", "google/*", "config.json", "Wan2.1_VAE.pth",
                        "models_t5_umt5-xxl-enc-bf16.pth", "README.md"])
    manifest = json.loads(Path("/root/lingbot-weights.json").read_text())
    bad = []
    for item in manifest["files"]:
        path = Path(target) / item["path"]
        if not path.exists() or path.stat().st_size != item["bytes"]:
            bad.append(item["path"])
    weights.commit()
    return {"files": len(manifest["files"]), "bad": bad, "revision": manifest["revision"]}


@app.local_entrypoint()
def main():
    print(json.dumps(download.remote(), indent=2))
