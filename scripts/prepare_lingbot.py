"""Download the official, unquantized 14B LingBot-World2 causal-fast release."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from prepare_matrix import fetch, valid

ROOT = Path(__file__).resolve().parents[1]
REPO = "robbyant/lingbot-world-v2-14b-causal-fast"
REVISION = "5c33dd40b213598c418fd25bff30fdbd23fd38a7"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8, help="Concurrent files, including all eight transformer shards")
    parser.add_argument("--range-workers", type=int, default=16, help="Concurrent HTTP ranges per resumed/fallback file")
    parser.add_argument("--transport", choices=["auto", "range"], default="auto")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or not 1 <= args.range_workers <= 64:
        parser.error("invalid transfer concurrency")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    hf_download = None
    if args.transport == "auto":
        try:
            import requests
            import hf_xet
            from huggingface_hub import hf_hub_download, configure_http_backend
            def backend():
                session = requests.Session()
                session.headers["Accept-Encoding"] = "identity"
                return session
            configure_http_backend(backend_factory=backend)
            hf_download = hf_hub_download
        except ImportError:
            print("Xet unavailable; using parallel HTTP ranges", flush=True)
    url = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true"
    with urlopen(Request(url, headers={"Accept-Encoding": "identity"}), timeout=60) as response:
        metadata = json.load(response)
    if metadata["sha"] != REVISION:
        raise ValueError("checkpoint revision mismatch")
    entries = [entry for entry in metadata["siblings"] if entry["rfilename"].startswith(("transformers/", "google/"))
               or entry["rfilename"] in {"config.json", "Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "README.md"}]
    root = ROOT / "external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast"
    reuse = ROOT / "external/Matrix-Game/checkpoints/Matrix-Game-3.0"
    print(f"Preparing {len(entries)} files ({sum(f['size'] for f in entries) / 1e9:.2f} GB) from {REPO}@{REVISION}", flush=True)
    print(f"Concurrent files={args.workers}; resumed/fallback ranges per file={args.range_workers}; Xet high performance={hf_download is not None}", flush=True)
    def transfer(entry):
        name = entry["rfilename"]
        path = root / name
        # Keep all verified data and resume existing HTTP shards. Fresh files
        # use HF's Rust Xet client, with high-performance adaptive concurrency.
        parts = list(path.parent.glob(path.name + ".part.*"))
        if not valid(path, entry) and hf_download is not None and not parts and entry["size"] > 512 * 1024 * 1024:
            print(f"Xet starting {name}", flush=True)
            try:
                hf_download(repo_id=REPO, filename=name, revision=REVISION, local_dir=root)
                if not valid(path, entry):
                    raise ValueError("Xet result failed upstream checksum verification")
                print(f"Xet downloaded and verified {name}", flush=True)
                return {"path": name, "bytes": entry["size"], "sha256": entry["lfs"]["sha256"]}
            except Exception as error:
                print(f"Xet fallback for {name}: {type(error).__name__}", flush=True)
        return fetch(entry, root, reuse, repo=REPO, revision=REVISION, range_workers=args.range_workers)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        files = list(pool.map(transfer, entries))
    output = ROOT / "reproduction/lingbot-weights.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps({"repository": REPO, "revision": REVISION, "variant": "14B causal-fast",
        "quantization": "none", "paper_checkpoint_revision_verified": False,
        "transfer": {"concurrent_files": args.workers, "range_workers_per_file": args.range_workers,
                     "xet_high_performance": hf_download is not None}, "files": files}, indent=2) + "\n")
    temporary.replace(output)
    print(f"Verified checkpoint inventory: {output.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
