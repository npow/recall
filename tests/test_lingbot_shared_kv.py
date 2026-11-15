from types import SimpleNamespace

import numpy as np
import pytest
import torch

from worldsync.lingbot_controls import CameraPlan
from worldsync.lingbot_shared_kv import Frame, SharedPoseArchive, SharedPoseKVMemory, splice_peer_kv


def pose(degrees=0):
    angle = np.deg2rad(degrees)
    p = np.eye(4)
    p[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    return p


def frame(actor, index, degrees=0, revision=0, source="visit"):
    kv = torch.zeros(2, 2, 1, 1, 1)
    return Frame(f"{actor}:{source}:{index}", actor, source, index, pose(degrees), revision,
                 (1, 1, 1, 1, 1), kv)


def test_only_pose_matched_peer_commits_of_same_revision_are_retrieved():
    archive = SharedPoseArchive()
    archive.commit([frame("A", 8, 60), frame("B", 8, 60), frame("B", 9, -60), frame("B", 10, 60, revision=1)])
    selected = archive.select("A", pose(60))
    assert [f.id for f in selected] == ["B:visit:8"]
    assert not archive.select("A", pose(180))
    assert not archive.select("A", pose(60), geometry=(2, 2, 4, 1, 1))


def test_eviction_preserves_peer_left_and_right_views_for_actual_paths():
    archive = SharedPoseArchive(frames_per_actor=8)
    for actor, actions in [("B", ["WAIT", "RIGHT", "WAIT"]),
                           ("A", ["WAIT", "LEFT", "WAIT", "RIGHT", "RIGHT", "WAIT"])]:
        camera = CameraPlan()
        for chunk, action in enumerate(actions):
            poses = camera.advance(action)
            rows = []
            for index in range(chunk*4, chunk*4+4):
                if index < 6:
                    continue
                f = frame(actor, index)
                f.pose = poses[index]
                rows.append(f)
            archive.commit(rows)
    assert archive.select("A", pose(60))
    assert archive.select("B", pose(-60))
    assert len(archive.frames) <= 16


def test_real_splice_includes_peer_values_and_preserves_current_queries():
    native = torch.arange(18).view(1, 18, 1, 1).float()
    peer = torch.full((1, 4, 1, 1), 99.)
    k, v = splice_peer_kv(native, native+100, peer, peer+100, 1)
    assert k.flatten().tolist() == [0, 1, 2, 3, 4, 5, 99, 99, 99, 99, 10, 11, 12, 13, 14, 15, 16, 17]
    assert v.flatten().tolist() == (k+100).flatten().tolist()
    assert torch.equal(k[:, -4:], native[:, -4:])
    assert k.shape == native.shape


@pytest.fixture
def memory(monkeypatch):
    def parent(self, pipeline, **kwargs):
        self.active, self.raw_key, self.clean = None, None, False
        self.cached, self.trace, self.retrieval_calls = {}, [], 0
        self.original_attention = lambda q, k, v, **kw: (k.clone(), v.clone())
        self.module = SimpleNamespace(causal_rope_apply=lambda keys, *a, **kw: keys)
    monkeypatch.setattr("worldsync.lingbot_shared_kv.PoseKVMemory.__init__", parent)
    archive = SharedPoseArchive()
    peer = frame("B", 8)
    peer.kv[:, 0] = 99
    peer.kv[:, 1] = 199
    archive.commit([peer])
    pipeline = SimpleNamespace(model=SimpleNamespace(blocks=[None, None]))
    return SharedPoseKVMemory(pipeline, archive, "A", "test")


def test_attention_reads_foreign_buffer_and_logs_source_actor(memory):
    memory.begin(4, np.stack([pose()]*20))
    memory.active = (0, 16, 1, torch.tensor([[4, 1, 1]]), None)
    native = torch.arange(18).view(1, 18, 1, 1).float()
    keys, values = memory.attend(torch.zeros(1, 4, 1, 1), native, native+100)
    assert keys[0, 6, 0, 0] == 99 and values[0, 6, 0, 0] == 199
    assert memory.trace[-1]["peer_frames"][0]["actor"] == "B"
    assert memory.trace[-1]["layers_injected"] == [0]
    assert memory.trace[-1]["attention_replaced"]


def test_partial_layer_capture_is_invisible_until_atomic_commit(memory):
    memory.begin(2, np.stack([pose()]*12))
    memory.clean = True
    memory.raw_key = torch.ones(1, 4, 1, 1)
    native = torch.zeros(1, 12, 1, 1)
    memory.active = (0, 8, 1, torch.tensor([[4, 1, 1]]), None)
    memory.attend(memory.raw_key, native, native)
    assert not memory.archive.select("B", pose())
    with pytest.raises(RuntimeError, match="incomplete"):
        memory.observe(2, None)
    memory.active = (1, 8, 1, torch.tensor([[4, 1, 1]]), None)
    memory.attend(memory.raw_key, native, native)
    memory.observe(2, None)
    assert memory.archive.select("B", pose())
    assert not memory.pending


def test_no_matching_peer_is_stock_attention(memory):
    memory.begin(4, np.stack([pose(180)]*20))
    memory.active = (0, 16, 1, torch.tensor([[4, 1, 1]]), None)
    native = torch.randn(1, 18, 1, 1)
    keys, values = memory.attend(torch.zeros(1, 4, 1, 1), native, native)
    assert torch.equal(keys, native) and torch.equal(values, native)
    assert not memory.trace[-1]["attention_replaced"]
