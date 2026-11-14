"""Fetch the pinned stock Matrix release; verify all files, reuse identical weights."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO = "Skywork/Matrix-Game-3.0"
REVISION = "d4c84ef2b8e9028a00a5d58b6be2df04f1cb6042"


def digest(path, kind="sha256", git_blob=False):
    result = hashlib.new(kind)
    if git_blob:
        result.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def valid(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    if entry.get("lfs"):
        return digest(path) == entry["lfs"]["sha256"]
    return digest(path, "sha1", git_blob=True) == entry["blobId"]


def parallel_download(url, temporary, size):
    """Bounded range downloads avoid serial CDN throttling; validate every range."""
    unit = 128 * 1024 * 1024
    parts = [(start, min(start + unit, size) - 1) for start in range(0, size, unit)]

    def fetch_part(bounds):
        start, end = bounds
        part = temporary.with_name(f"{temporary.name}.{start}-{end}")
        expected = end - start + 1
        for attempt in range(5):
            offset = part.stat().st_size if part.exists() else 0
            if offset == expected:
                return part
            if offset > expected:
                part.unlink()
                offset = 0
            headers = {"Accept-Encoding": "identity", "Range": f"bytes={start + offset}-{end}"}
            try:
                # Include the range in the URL to avoid intermediaries caching a
                # different partial response for the same URL.
                ranged_url = url + f"?range_start={start + offset}&range_end={end}"
                with urlopen(Request(ranged_url, headers=headers), timeout=120) as response:
                    wanted = f"bytes {start + offset}-{end}/{size}"
                    if response.status != 206 or response.headers.get("Content-Range") != wanted:
                        raise ValueError("server did not honor requested byte range")
                    with part.open("ab") as output:
                        while block := response.read(8 * 1024 * 1024):
                            output.write(block)
                if part.stat().st_size != expected:
                    raise ValueError("incomplete byte range")
                return part
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(attempt + 1)

    with ThreadPoolExecutor(max_workers=24) as pool:
        paths = list(pool.map(fetch_part, parts))
    with temporary.open("wb") as output:
        for part in paths:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    # Keep shards until whole-file integrity has been checked by the caller.
    return paths


def fetch(entry, root, reuse_root):
    name = entry["rfilename"]
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not valid(path, entry):
        reuse = reuse_root / name
        if valid(reuse, entry):
            # Keep checkpoints independent of the exploratory ABot checkout.
            shutil.copyfile(reuse, path)
            print(f"Reused checksum-identical {name}", flush=True)
        else:
            temporary = path.with_suffix(path.suffix + ".part")
            url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{quote(name, safe='/')}"
            if entry["size"] > 512 * 1024 * 1024:
                parts = parallel_download(url, temporary, entry["size"])
                if not valid(temporary, entry):
                    raise ValueError(f"checksum mismatch: {name}; remove range shards before retrying")
                temporary.replace(path)
                for part in parts:
                    part.unlink()
                print(f"Downloaded and verified {name}", flush=True)
                return {"path": name, "bytes": entry["size"], "sha256": digest(path)}
            for attempt in range(5):
                try:
                    offset = temporary.stat().st_size if temporary.exists() else 0
                    headers = {"Accept-Encoding": "identity"}
                    if offset:
                        headers["Range"] = f"bytes={offset}-"
                    with urlopen(Request(url, headers=headers), timeout=120) as response:
                        append = response.status == 206 and offset > 0
                        if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise ValueError("invalid resume range")
                        with temporary.open("ab" if append else "wb") as output:
                            while block := response.read(8 * 1024 * 1024):
                                output.write(block)
                    if not valid(temporary, entry):
                        temporary.unlink(missing_ok=True)
                        raise ValueError(f"checksum mismatch: {name}")
                    temporary.replace(path)
                    break
                except Exception as error:
                    print(f"Retry {attempt + 1} {name}: {type(error).__name__}", flush=True)
                    if attempt == 4:
                        raise
                    time.sleep(attempt + 1)
            print(f"Downloaded and verified {name}", flush=True)
    return {"path": name, "bytes": entry["size"], "sha256": digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("external/Matrix-Game/checkpoints/Matrix-Game-3.0"))
    parser.add_argument("--reuse-root", type=Path, default=Path("external/ABot-World/checkpoints/ABot-World-0-5B-LF"))
    args = parser.parse_args()
    url = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true"
    with urlopen(Request(url, headers={"Accept-Encoding": "identity"}), timeout=60) as response:
        metadata = json.load(response)
    if metadata["sha"] != REVISION:
        raise ValueError("checkpoint revision mismatch")
    entries = [entry for entry in metadata["siblings"] if entry["rfilename"].startswith(("base_distilled_model/", "google/"))
               or entry["rfilename"] in {"Wan2.2_VAE.pth", "MG-LightVAE.pth", "MG-LightVAE_v2.pth", "models_t5_umt5-xxl-enc-bf16.pth", "README.md", "model_index.json"}]
    print(f"Preparing {len(entries)} files from {REPO}@{REVISION}", flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(lambda entry: fetch(entry, args.root, args.reuse_root), entries))
    output = Path("reproduction/matrix-weights.json")
    output.write_text(json.dumps({"repository": REPO, "revision": REVISION,
        "paper_checkpoint_revision_verified": False, "files": files}, indent=2) + "\n")
    print(f"Verified checkpoint inventory: {output}", flush=True)


if __name__ == "__main__":
    main()
