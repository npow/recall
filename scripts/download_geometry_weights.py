"""Parallel, resumable download of the pinned pretrained DA3-Giant geometry model."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
REVISION = "72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19"
SIZE = 5422814644
SHA256 = "1e47a08338ca73a6d6a21d37fd060b26b993b672bc6ddf6295fe474df2592001"
URL = f"https://huggingface.co/depth-anything/DA3-GIANT-1.1/resolve/{REVISION}/model.safetensors"
DIRECTORY = ROOT / "external/Depth-Anything-3/checkpoints/DA3-GIANT-1.1"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = DIRECTORY / "config.json"
    config_sha = "74626a50d6dee2a11820291a4305c1a34aa5adc4f7260908bbdbc9a939ba8e93"
    if not config.exists() or digest(config) != config_sha:
        response = requests.get(URL.rsplit("/", 1)[0] + "/config.json", timeout=(15, 30))
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != config_sha:
            raise ValueError("Pinned config hash mismatch")
        config.write_bytes(response.content)
    target = DIRECTORY / "model.safetensors"
    if target.exists() and target.stat().st_size == SIZE and digest(target) == SHA256:
        print("Pinned geometry weights already verified", flush=True)
        return
    parts = DIRECTORY / ".range-download"
    parts.mkdir(exist_ok=True)
    chunk = 128 * 1024 * 1024
    ranges = [(index, start, min(start + chunk, SIZE) - 1) for index, start in enumerate(range(0, SIZE, chunk))]

    def fetch(item):
        index, start, end = item
        part = parts / f"{index:03d}.part"
        expected = end - start + 1
        if part.exists() and part.stat().st_size == expected:
            return index
        for attempt in range(3):
            offset = part.stat().st_size if part.exists() else 0
            if offset > expected:
                raise ValueError("Oversized partial download")
            try:
                # Distinct cache keys prevent proxies from replaying another
                # range. Validate the response before writing a single byte.
                with requests.get(URL, params={"download": "true", "part": index, "offset": offset},
                                  headers={"Range": f"bytes={start + offset}-{end}", "Accept-Encoding": "identity"},
                                  stream=True, timeout=(15, 60)) as response:
                    expected_range = f"bytes {start + offset}-{end}/{SIZE}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                        raise ValueError(f"Unexpected range response for part{index}: status{response.status_code}")
                    with part.open("ab") as destination:
                        for block in response.iter_content(1024 * 1024):
                            destination.write(block)
                if part.stat().st_size != expected:
                    raise ValueError("Incomplete range body")
                return index
            except (requests.RequestException, ValueError):
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(fetch, item) for item in ranges]):
            print(f"Verified range length for part{future.result()} in{time.monotonic() - started:.1f}s", flush=True)
    temporary = DIRECTORY / "model.safetensors.assembling"
    with temporary.open("wb") as destination:
        for index, _, _ in ranges:
            with (parts / f"{index:03d}.part").open("rb") as source:
                for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    destination.write(block)
    if temporary.stat().st_size != SIZE or digest(temporary) != SHA256:
        raise ValueError("Pinned complete-file SHA256 verification failed; retain partials")
    temporary.replace(target)
    print(f"Verified complete pinned model: {SIZE} bytes, SHA256{SHA256}", flush=True)


if __name__ == "__main__":
    main()
