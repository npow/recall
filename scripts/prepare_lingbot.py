"""Download the official, unquantized 14B LingBot-World2 causal-fast release."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib.request import Request, urlopen

from prepare_matrix import fetch

ROOT = Path(__file__).resolve().parents[1]
REPO = "robbyant/lingbot-world-v2-14b-causal-fast"
REVISION = "5c33dd40b213598c418fd25bff30fdbd23fd38a7"


def main():
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
    with ThreadPoolExecutor(max_workers=2) as pool:
        files = list(pool.map(lambda entry: fetch(entry, root, reuse, repo=REPO, revision=REVISION), entries))
    output = ROOT / "reproduction/lingbot-weights.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps({"repository": REPO, "revision": REVISION, "variant": "14B causal-fast",
        "quantization": "none", "paper_checkpoint_revision_verified": False, "files": files}, indent=2) + "\n")
    temporary.replace(output)
    print(f"Verified checkpoint inventory: {output.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
