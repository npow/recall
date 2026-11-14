import hashlib
import json

from PIL import Image
import pytest

from worldsync import repro_assets


def test_source_verification_detects_same_size_corruption(tmp_path):
    path = tmp_path / "asset"
    data = b"original"
    path.write_bytes(data)
    sources = [
        {"size": len(data), "lfs": {"sha256": hashlib.sha256(data).hexdigest()}},
        {"size": len(data), "blobId": hashlib.sha1(b"blob 8\0" + data).hexdigest()},
    ]
    for source in sources:
        assert repro_assets.verify_source(path, source)
    path.write_bytes(b"modified")
    for source in sources:
        assert not repro_assets.verify_source(path, source)


def test_inventory_rejects_missing_images_duplicate_cases_and_escaped_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(repro_assets, "SPLITS", {"gc": 1})
    folder = tmp_path / "gc"
    folder.mkdir()
    metadata = folder / "data.json"
    record = {"task_id": "GC001", "prompt": "Walk", "image_caption": "A path",
              "questions": ["Is the path stable?"], "image_path": "gc/input.png"}
    metadata.write_text(json.dumps([record]))
    with pytest.raises(ValueError, match="missing image"):
        repro_assets.inventory_cases(tmp_path)
    Image.new("RGB", (8, 8)).save(folder / "input.png")
    assert len(repro_assets.inventory_cases(tmp_path)) == 1
    monkeypatch.setattr(repro_assets, "SPLITS", {"gc": 2})
    metadata.write_text(json.dumps([record, record]))
    with pytest.raises(ValueError, match="duplicate"):
        repro_assets.inventory_cases(tmp_path)
    monkeypatch.setattr(repro_assets, "SPLITS", {"gc": 1})
    record["image_path"] = "../outside.png"
    metadata.write_text(json.dumps([record]))
    with pytest.raises(ValueError, match="escapes"):
        repro_assets.inventory_cases(tmp_path)
