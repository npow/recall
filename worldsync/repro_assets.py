"""Fetch and inventory immutable benchmark inputs and selected published videos.

This prepares reproduction assets. It does not generate or grade a world, and it
does not certify reproduction of a published score.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import quote
from urllib.request import Request, urlopen


DATASET = "jocelynd/playworld-bench"
REVISION = "697e177194a1c4d8afe9068756fe743ac23ee1e8"
SPLITS = {"gc": 48, "if": 50, "insight": 30, "outsight": 43}
REFERENCES = [
    {"model": "Genie 3", "split": "gc", "task_id": "GC004", "filename": "genie3-gc004.mp4"},
    {"model": "Matrix-Game 3.0", "split": "gc", "task_id": "GC007", "filename": "matrixgame3-gc007.mp4"},
    {"model": "Matrix-Game 3.0", "split": "insight", "task_id": "OE001", "filename": "matrixgame3-oe001.mp4"},
]


def safe_path(root: Path, name: str):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"asset path escapes download directory: {name}")
    return path


def verify_source(path, expected):
    data = path.read_bytes()
    if len(data) != expected["size"]:
        return False
    if expected.get("lfs"):
        return hashlib.sha256(data).hexdigest() == expected["lfs"]["sha256"]
    blob = b"blob " + str(len(data)).encode() + b"\0" + data
    return hashlib.sha1(blob).hexdigest() == expected["blobId"]


def download(url, path, force=False, expected=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if expected and path.is_file() and not verify_source(path, expected):
        force = True
    if force or not path.is_file():
        temporary = path.with_suffix(path.suffix + ".part")
        for attempt in range(3):
            try:
                request = Request(url, headers={"Accept-Encoding": "identity", "User-Agent": "worldsync-reproduction/0.1"})
                with urlopen(request, timeout=90) as response, temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                if expected and not verify_source(temporary, expected):
                    raise ValueError(f"source checksum mismatch: {path.name}")
                temporary.replace(path)
                break
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def inventory_cases(root: Path, require_images=True):
    cases = []
    for split, expected in SPLITS.items():
        records = json.loads((root / split / "data.json").read_text())
        if not isinstance(records, list) or len(records) != expected:
            raise ValueError(f"{split}: expected {expected} cases")
        seen = set()
        for record in records:
            task_id = record["task_id"]
            if task_id in seen:
                raise ValueError(f"{split}: duplicate task {task_id}")
            seen.add(task_id)
            if not record.get("prompt") or not record.get("image_caption") or not record.get("questions"):
                raise ValueError(f"{split}/{task_id}: missing objective, caption, or rubric")
            image = safe_path(root, record["image_path"])
            if require_images and not image.is_file():
                raise ValueError(f"missing image: {record['image_path']}")
            case = {"split": split, "task_id": task_id, "source_task_id": record.get("source_task_id"),
                    "image_path": record["image_path"], "questions": len(record["questions"]),
                    "perspective": record.get("perspective"), "image_present": image.is_file()}
            if image.is_file():
                from PIL import Image
                with Image.open(image) as opened:
                    opened.verify()
                case["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
                case["image_bytes"] = image.stat().st_size
            cases.append(case)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("external/PlayWorld/data"))
    parser.add_argument("--reference-root", type=Path, default=Path("external/published-reference-videos"))
    parser.add_argument("--output", type=Path, default=Path("reproduction/assets.json"))
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--with-reference-videos", action="store_true")
    args = parser.parse_args()
    root = args.data_root.resolve()
    base = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/"
    url = f"https://huggingface.co/api/datasets/{DATASET}/revision/{REVISION}?blobs=true"
    with urlopen(Request(url, headers={"Accept-Encoding": "identity"}), timeout=60) as response:
        source = json.load(response)
    if source["sha"] != REVISION:
        raise ValueError("dataset revision does not match the frozen reference")
    source_files = {entry["rfilename"]: entry for entry in source["siblings"]}
    metadata = {}
    for name in ["README.md", *[f"{split}/data.json" for split in SPLITS]]:
        metadata[name] = download(base + quote(name, safe="/"), safe_path(root, name), expected=source_files[name])
    cases = inventory_cases(root, require_images=False)
    if not args.metadata_only:
        names = sorted({case["image_path"] for case in cases})
        def fetch_image(name):
            return name, download(base + quote(name, safe="/"), safe_path(root, name), expected=source_files[name])
        with ThreadPoolExecutor(max_workers=6) as pool:
            for index, _ in enumerate(pool.map(fetch_image, names), 1):
                if index % 25 == 0 or index == len(names):
                    print(f"Downloaded/checked {index}/{len(names)} input images", flush=True)
        cases = inventory_cases(root)
    references = []
    if args.with_reference_videos:
        for reference in REFERENCES:
            if (reference["split"], reference["task_id"]) not in {(c["split"], c["task_id"]) for c in cases}:
                raise ValueError("reference case is absent from pinned dataset")
            url = "https://kxding.github.io/project/PlayWorld/demos/" + reference["filename"]
            path = safe_path(args.reference_root, reference["filename"])
            info = download(url, path)
            probe = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
                "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration", "-of", "json", str(path)], text=True)
            references.append({**reference, **info, "source_url": url, "ffprobe": json.loads(probe),
                "scope": "selected public demo; editing/re-encoding not independently verified; not a complete evaluation archive"})
            print(f"Inventoried published reference: {reference['filename']}", flush=True)
    manifest = {"status": "assets_only_not_reproduced", "dataset": DATASET, "dataset_revision": REVISION,
                "case_counts": SPLITS, "total_cases": len(cases), "input_images_complete": all(c["image_present"] for c in cases),
                "input_images_verified_against_source": not args.metadata_only,
                "metadata_files": metadata, "cases": cases, "reference_videos": references,
                "published_scores_reproduced": False, "hillclimbing_allowed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "cases": len(cases), "images_complete": manifest["input_images_complete"], "reference_videos": len(references)}))


if __name__ == "__main__":
    main()
