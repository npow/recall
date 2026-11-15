"""Verify the existing secondary-workspace checkpoint without another download."""
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("worldsync-lingbot-verify-weights")
weights = modal.Volume.from_name("worldsync-lingbot-weights")
image = modal.Image.debian_slim(python_version="3.12").add_local_file(
    ROOT / "reproduction/lingbot-weights.json", "/work/weights.json")


@app.function(image=image, cpu=4, memory=8192, timeout=1800, volumes={"/cache": weights})
def verify():
    from concurrent.futures import ThreadPoolExecutor
    import hashlib
    import time
    manifest = json.loads(Path("/work/weights.json").read_text())
    root = Path("/cache/14b-causal-fast")
    started = time.monotonic()
    for attempt in range(100):
        weights.reload()
        missing = [row["path"] for row in manifest["files"]
                   if not (root / row["path"]).exists() or (root / row["path"]).stat().st_size != row["bytes"]]
        if not missing:
            break
        print(f"Waiting for existing staging job: {len(missing)} incomplete files", flush=True)
        time.sleep(15)
    else:
        raise RuntimeError("Existing checkpoint staging has not completed; no GPU job started")

    def check(row):
        digest = hashlib.sha256()
        with (root / row["path"]).open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != row["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {row['path']}")
        return {"path": row["path"], "bytes": row["bytes"], "sha256": digest.hexdigest()}

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(check, manifest["files"]))
    return {"status": "verified", "revision": manifest["revision"], "files": files,
            "elapsed_s": time.monotonic() - started, "downloaded_bytes": 0}


@app.local_entrypoint()
def main(workspace: str):
    report = verify.remote()
    report["workspace"] = workspace
    output = ROOT / "outputs/lingbot-reproduction" / f"weights-verified-{workspace}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))
