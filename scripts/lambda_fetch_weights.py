"""Download the pinned LingBot-World2 14B causal-fast weights and verify every file (run ON the instance).

    cd ~/work && .venv-lambda/bin/python scripts/lambda_fetch_weights.py

Uses the pinned revision and the size + SHA-256 manifest in reproduction/lingbot-weights.json; any mismatch aborts.
"""
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "reproduction/lingbot-weights.json").read_text())
target = ROOT / "external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast"
target.mkdir(parents=True, exist_ok=True)


def fetch(item):
    path = hf_hub_download(manifest["repository"], item["path"], revision=manifest["revision"], local_dir=target)
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    if os.path.getsize(path) != item["bytes"] or digest.hexdigest() != item["sha256"]:
        raise ValueError(f"Pinned weight verification failed: {item['path']}")
    return item["path"]


started = time.monotonic()
with ThreadPoolExecutor(max_workers=8) as pool:
    names = list(pool.map(fetch, manifest["files"]))
print(json.dumps({"verified_files": len(names), "revision": manifest["revision"],
                  "seconds": round(time.monotonic() - started, 1), "status": "WEIGHTS_VERIFIED"}))
