"""Offline tests for the shared-world memory (local SQLite frames, past.dev refs; no network, no key)."""
import io
import json
import shutil
import subprocess

import pytest

from worldsync import shared_world as sw
from tests.test_research_memory import client


def run_json(seed=42, pixels=("p0", "p1", "p2")):
    headings = [0.0, 45.0, 90.0]
    return {"model": "m", "source_revision": "s", "checkpoint_revision": "c", "task_id": "GC010",
            "initial_image_sha256": "i" * 64, "image_caption_sha256": "t" * 64, "seed": seed, "fps": 16, "chunk_size": 4,
            "chunks": [{"index": i, "frames": 13 if i == 0 else 16, "pixel_sha256": p} for i, p in enumerate(pixels)],
            "actions": [{"action": a, "pose_after": {"heading_degrees": h, "world_time_seconds": 0.75 + i,
                                                     "position_normalized_units": [0.0, 0.0, 0.0]}}
                        for i, (a, h) in enumerate(zip(["WAIT", "RIGHT", "RIGHT"], headings))]}


def test_world_id_depends_on_start_conditions_not_memory_method():
    a, b = sw.world_manifest(run_json()), sw.world_manifest(run_json())
    b["model"], b["task_id"] = "other memory method", "other"
    assert sw.world_id(a) == sw.world_id(b)
    for key, value in [("seed", 43), ("fps", 30), ("chunk_size", 3), ("initial_image_sha256", "j" * 64)]:
        assert sw.world_id({**a, key: value}) != sw.world_id(a), key
    with pytest.raises(ValueError):
        sw.world_id({**a, "initial_image_sha256": None})


def test_sightings_use_last_frame_of_each_chunk_and_pose_after():
    sightings = sw.run_sightings(run_json())
    assert [s["frame_index"] for s in sightings] == [12, 28, 44]
    assert [s["heading"] for s in sightings] == [0.0, 45.0, 90.0]


def test_heading_gap_wraps():
    assert sw.heading_gap(350, 10) == 20 and sw.heading_gap(10, 350) == 20 and sw.heading_gap(0, 180) == 180


def test_frame_db_dedupes_and_finds_nearby():
    db = sw.FrameDB(":memory:")
    digest = db.put_frame(b"jpeg-bytes")
    assert db.put_frame(b"jpeg-bytes") == digest and db.get_frame(digest) == b"jpeg-bytes"
    for agent, chunk, heading, t in [("a", 0, 355, 1.0), ("a", 1, 90, 2.0), ("b", 0, 5, 3.0)]:
        db.add_sighting("w", agent, {"chunk": chunk, "frame_index": chunk, "world_time": t, "heading": heading,
                                     "position": [0, 0, 0], "frame_sha256": digest})
    assert [(s["agent"], s["heading"]) for s in db.nearby("w", 0, 10)] == [("b", 5), ("a", 355)]  # tie: newest first
    assert [s["agent"] for s in db.nearby("w", 0, 10, exclude_agent="a")] == ["b"]
    assert [s["agent"] for s in db.nearby("w", 0, 10, before=2.0)] == ["a"]


def test_observation_point_is_text_with_frame_reference():
    sighting = {**sw.run_sightings(run_json())[2], "frame_sha256": "f" * 64}
    point = sw.observation_point("w-1", "agent-a", sighting, audience="agent-a-private", thumbnail="dGh1bWI=")
    assert "/" not in point["id"] and point["identity"] == "agent-a" and point["audience"] == "agent-a-private"
    assert f"frame:sha256:{'f' * 64}" in point["content"] and "heading 90" in point["content"]
    assert point["timestamp"] == "2026-01-01T00:00:02.750000Z"
    assert point["metadata"]["thumbnail_jpeg_b64"] == "dGh1bWI="
    assert len(point["content"]) < 32000
    assert "caption" not in point["content"] and " saw " not in point["content"]
    captioned = sw.observation_point("w-1", "agent-a", {**sighting, "caption": "the SHERIFF office"})
    assert "and saw the SHERIFF office. Frame reference" in captioned["content"]


def test_divergence_reports_first_differing_chunk():
    a, b = run_json(), run_json(pixels=("p0", "p1", "x2"))
    assert sw.divergence(a, b) == 2 and sw.divergence(a, run_json()) is None
    point = sw.divergence_point("w-1", ("agent-a", "agent-b"), a, b)
    assert "through chunk 1" in point["content"] and point["metadata"]["first_divergent_chunk"] == 2
    assert point["timestamp"] == sw.world_timestamp(1.75)
    first = sw.divergence_point("w-1", ("agent-a", "agent-b"), a, run_json(pixels=("x0", "p1", "p2")))
    assert first["metadata"]["first_divergent_chunk"] == 0 and "from the first chunk" in first["content"]
    assert first["timestamp"] == sw.world_timestamp(0)
    assert sw.divergence(run_json(pixels=(None, None, None)), run_json(pixels=(None, None, None))) == 0
    short = run_json(pixels=("p0", "p1"))
    short["actions"] = short["actions"][:2]
    same = sw.divergence_point("w-1", ("agent-a", "agent-b"), a, short)
    assert "all 2 compared chunks" in same["content"] and same["timestamp"] == sw.world_timestamp(1.75)


def test_outbox_is_idempotent_and_dry_run_sends_nothing():
    db = sw.FrameDB(":memory:")
    point = sw.manifest_point("w-1", sw.world_manifest(run_json()))
    assert db.queue(point) and not db.queue(point)
    c, op = client([])
    report = sw.send_pending(db, client=c, dry_run=True, log=lambda *_: None)
    assert report["pending"] == 1 and report["sent"] == 0 and not op.requests
    assert report["credits"] == sw.ingest_credits(point) == -(-len(point["content"].encode()) // 350)
    assert sw.ingest_credits({"content": "x", "metadata": {"thumbnail_jpeg_b64": "y" * 5000}}) == 1
    c, op = client([{"ingestionId": "i1", "status": "queued"}])
    assert sw.send_pending(db, client=c, dry_run=False, log=lambda *_: None)["sent"] == 1
    assert json.loads(op.requests[0].data)["id"] == "world:w-1:manifest" and not db.pending()


def test_recall_view_resolves_frame_refs_locally():
    db = sw.FrameDB(":memory:")
    digest = db.put_frame(b"jpeg")
    missing = "0" * 64
    page = {"results": [{"rank": 1, "occurredAt": "2026-01-01T00:00:02Z", "content": f"saw frame:sha256:{digest}",
                         "sources": [{"metadata": {"world_id": "w-1", "frame_ref": f"frame:sha256:{missing}"},
                                      "excerpts": []}]},
                        {"rank": 2, "occurredAt": "2026-01-01T00:00:02Z", "content": "another world",
                         "sources": [{"metadata": {"world_id": "w-2"}, "excerpts": []}]}]}
    c, op = client([page])
    out = sw.recall_view("w-1", 90, "agent-b", db=db, client=c, at=2.75)
    body = json.loads(op.requests[0].data)
    assert body["identity"] == "agent-b" and body["queryTimestamp"] == "2026-01-01T00:00:02.750000Z"
    assert out["results"][0]["frames"] == sorted([digest, missing])
    assert out["results"][0]["frames_available"] == [digest]
    assert [r["rank"] for r in out["results"]] == [1]


def test_cli_requires_exactly_one_send_mode(capsys):
    for flags in ([], ["--dry-run", "--confirm-send"]):
        with pytest.raises(SystemExit):
            sw.main(["--db", ":memory:", "compare", "a", "b", "--agents", "x", "y", *flags])


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_frames_from_video_decodes_in_memory(tmp_path):
    video = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x32:rate=16", "-frames:v", "20",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    frames = sw.frames_from_video(video, [3, 12, 12])
    assert sorted(frames) == [3, 12]
    jpeg, width, height = frames[12]
    assert jpeg[:2] == b"\xff\xd8" and (width, height) == (64, 32)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["v.mp4"]
    assert sw.thumbnail_b64(jpeg, width=32)
