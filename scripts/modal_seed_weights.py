"""Seed a Modal workspace's worldsync-lingbot-weights volume from Hugging Face (public repo).

Run once per workspace:  MODAL_PROFILE=<profile> modal run scripts/modal_seed_weights.py
Layout matches the original volume: /14b-causal-fast/{transformers,google,...}.
"""
import modal

app = modal.App("worldsync-seed-weights")
weights = modal.Volume.from_name("worldsync-lingbot-weights", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub[hf_transfer]==0.30.2")


@app.function(image=image, volumes={"/cache": weights}, timeout=7200, cpu=8, memory=32768,
              env={"HF_HUB_ENABLE_HF_TRANSFER": "1"})
def seed(repo: str = "robbyant/lingbot-world-v2-14b-causal-fast", target: str = "14b-causal-fast"):
    import os
    from huggingface_hub import snapshot_download
    path = f"/cache/{target}"
    os.makedirs(path, exist_ok=True)
    snapshot_download(repo_id=repo, local_dir=path, max_workers=16)
    weights.commit()
    total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs
                if ".cache" not in d)
    return {"path": path, "bytes": total, "files": sorted(os.listdir(path))}


@app.local_entrypoint()
def main():
    print(seed.remote())
