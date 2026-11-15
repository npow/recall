from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest

from worldsync.shared_world import INITIAL, World, render_context
from worldsync.multi_agent_demo import publish_past, run


def make_world(tmp_path):
    world = World(tmp_path / "world.sqlite")
    world.initialize()
    return world


def test_stale_write_does_not_mutate_world_and_retry_is_idempotent(tmp_path):
    world = make_world(tmp_path)
    first = world.submit("alice", "move", 0, "one", target=[2, 2])
    assert first["accepted"]
    stale = world.submit("bob", "move", 0, "two", target=[6, 2])
    assert not stale["accepted"]
    assert world.read() == first["state"]
    assert world.submit("alice", "move", 0, "one", target=[2, 2]) == first
    assert len(world.history()) == 2
    with pytest.raises(ValueError, match="retry token"):
        world.submit("alice", "pickup", 1, "one")
    assert world.read() == first["state"]


def test_concurrent_pickup_has_exactly_one_winner(tmp_path):
    world = make_world(tmp_path)
    world.submit("alice", "move", 0, "a", target=[2, 2])
    for rev, x in enumerate(range(6, 1, -1), 1):
        assert world.submit("bob", "move", rev, f"b{x}", target=[x, 2])["accepted"]
    revision = world.read()["revision"]
    barrier = threading.Barrier(2)

    def pickup(actor):
        barrier.wait(timeout=5)
        return World(world.path).submit(actor, "pickup", revision, actor+"-pickup")

    with ThreadPoolExecutor(2) as pool:
        replies = list(pool.map(pickup, ["alice", "bob"]))
    assert sum(r["accepted"] for r in replies) == 1
    winner = next(r["actor"] for r in replies if r["accepted"])
    assert world.read()["parcel"] == {"id": "red-parcel", "position": None, "holder": winner}
    assert world.read()["revision"] == revision+1


def test_invalid_actions_preserve_state(tmp_path):
    world = make_world(tmp_path)
    attempts = [("move", {"target": [8, 2]}), ("move", {"target": [True, 3]}),
                ("pickup", {}), ("drop", {}), ("handoff", {"recipient": "bob"})]
    for index, (action, args) in enumerate(attempts):
        assert not world.submit("alice", action, 0, str(index), **args)["accepted"]
    assert world.read() == INITIAL


def test_full_demo_processes_converge_and_exports_only_commits(tmp_path):
    report = run(tmp_path)
    assert report["agents_converged"]
    assert len(set(report["agent_pids"].values())) == 2
    assert (report["accepted"], report["rejected"]) == (10, 1)
    assert report["final_state"]["parcel"]["holder"] == "alice"
    assert (tmp_path / "demo.gif").stat().st_size > 1000
    findings = [json.loads(line) for line in (tmp_path / "past-outbox.jsonl").read_text().splitlines()]
    assert [f["metadata"]["revision"] for f in findings] == list(range(1, 11))
    assert len({f["idempotencyKey"] for f in findings}) == 10
    assert not (tmp_path / "past-receipts.jsonl").exists()
    stale = next(f for f in report["frames"] if (f["reply"] or {}).get("accepted") is False)
    assert stale["reply"]["expected_revision"] == 0
    assert stale["state"]["revision"] == 5
    assert stale["beliefs"]["bob"]["revision"] == 0

    class FakePast:
        def __init__(self):
            self.findings = []

        def ingest(self, finding):
            self.findings.append(finding)
            return {"ingestionId": finding["id"], "status": "queued"}

    client = FakePast()
    publish_past(tmp_path, client)
    assert client.findings == findings
    assert len((tmp_path / "past-receipts.jsonl").read_text().splitlines()) == 10
    context = render_context(report["final_state"], "bob")
    assert context["world_revision"] == 10
    assert "held by alice" in context["scene_text"]
