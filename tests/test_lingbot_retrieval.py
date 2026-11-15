import numpy as np
import pytest

from worldsync.lingbot_paths import PATHS
from worldsync.lingbot_controls import CameraPlan
from worldsync.lingbot_retrieval import select_frames


def test_retrieval_uses_only_old_non_sink_views_and_requires_camera_overlap():
    camera = CameraPlan()
    for action in PATHS["round_trip"]:
        camera.advance(action)
    poses = np.stack(camera.poses)
    selected = select_frames(poses, range(64), 60)
    assert len(selected) == 4
    assert all(6 <= index < 56 for index in selected)
    assert select_frames(poses[:32], range(6, 20), 28) == []  # Facing away.
    assert select_frames(poses[:8], range(8), 4) == []  # No future or sink leakage.


def test_diverse_retrieval_does_not_spend_all_slots_on_duplicate_hold_views():
    camera = CameraPlan()
    for action in PATHS["turn_wait_return"][:12]:
        camera.advance(action)
    poses = np.stack(camera.poses)
    nearest = select_frames(poses, range(6, 22), 44, count=8)
    diverse = select_frames(poses, range(6, 22), 44, count=8, diverse=True)
    def unique_views(indices):
        return len({tuple(np.round(poses[index].ravel(), 7)) for index in indices})
    assert len(nearest) == 8 and unique_views(nearest) == 2
    assert len(diverse) == unique_views(diverse) == 5


@pytest.mark.parametrize("method", ["retrieval", "retrieval_reader", "retrieval_native", "retrieval_eight", "retrieval_diverse", "retrieval_eight_diverse", "retrieval_context22", "retrieval_eight_context22", "retrieval_eight_context22_gated", "sink_rephase"])
def test_actual_native_attention_retains_stock_output_until_retrieval_and_bounds_memory(monkeypatch, method):
    import copy
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    import torch
    source = Path(__file__).resolve().parents[1] / "external/LingBot-World2"
    if not source.exists():
        pytest.skip("pinned upstream checkout not installed")
    existing = sys.modules.get("wan")
    if existing and not Path(existing.__file__).resolve().is_relative_to(source.resolve()):
        # ABot and LingBot both vendor a package named wan. Exercise this native
        # integration in a fresh interpreter rather than replacing live modules.
        import subprocess
        subprocess.run([sys.executable, "-m", "pytest", "-q",
            f"{__file__}::test_actual_native_attention_retains_stock_output_until_retrieval_and_bounds_memory[{method}]"], check=True)
        return
    monkeypatch.syspath_prepend(str(source))
    from wan.modules import model_fast
    from worldsync.matrix_attention import sdpa_attention
    from worldsync.lingbot_retrieval import PoseKVMemory
    monkeypatch.setattr(model_fast, "attention", sdpa_attention)
    torch.manual_seed(7)
    capacity = 22 if "context22" in method else 18
    tokens = capacity * 4
    original = model_fast.CausalWanSelfAttention(64, 4, local_attn_size=capacity, sink_size=6).eval()
    candidate = copy.deepcopy(original)
    if method == "retrieval_reader":
        from worldsync.lingbot_memory_reader import MemoryReader
        candidate.memory_reader = MemoryReader(candidate.head_dim, rank=4)
    pipeline = SimpleNamespace(model=SimpleNamespace(blocks=[SimpleNamespace(self_attn=candidate)]))
    from worldsync.lingbot_sink_rephase import SinkRephaseMemory
    memory = (PoseKVMemory(pipeline, key_positions="native" if method == "retrieval_native" else "virtual",
                           retrieved_frames=8 if "eight" in method else 4,
                           selection="diverse" if "diverse" in method else "nearest", cache_frames=capacity,
                           history_weight=.25 if "gated" in method else None)
              if method.startswith("retrieval") else SinkRephaseMemory(pipeline))
    def cache():
        return {"k": torch.zeros(1, tokens, 4, 16), "v": torch.zeros(1, tokens, 4, 16),
                "global_end_index": torch.tensor([0]), "local_end_index": torch.tensor([0])}
    stock_cache, modified_cache = cache(), cache()
    frequency = torch.arange(1024)[:, None] * torch.linspace(.001, .1, 8)[None]
    freqs = torch.polar(torch.ones_like(frequency), frequency).to(torch.complex128)
    camera = CameraPlan()
    try:
        with torch.no_grad():
            name = "turn_wait_return" if any(token in method for token in ("eight", "diverse", "context22")) else ("round_trip" if method.startswith("retrieval") else "turn_return")
            for chunk, action in enumerate(PATHS[name]):
                camera.advance(action)
                memory.begin(chunk, camera.poses)
                memory.clean = True
                x = torch.randn(1, 16, 64)
                common = (x, torch.tensor([16]), torch.tensor([[4, 2, 2]]), freqs)
                expected = original(*common, stock_cache, chunk * 16, tokens, frame_seqlen=4)
                actual = candidate(*common, modified_cache, chunk * 16, tokens, frame_seqlen=4)
                assert torch.isfinite(actual).all()
                if not memory.trace[-1]["attention_replaced"]:
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                assert modified_cache["k"].shape == stock_cache["k"].shape
                assert len(memory.banks[0]) <= (16 if method.startswith("retrieval") else 6)
                if method == "retrieval_native" and chunk == 1:
                    torch.testing.assert_close(memory.banks[0][6][0], stock_cache["k"][:, 24:28], atol=0, rtol=0)
                    torch.testing.assert_close(memory.banks[0][7][0], stock_cache["k"][:, 28:32], atol=0, rtol=0)
                if method == "sink_rephase" or method.startswith("retrieval"):
                    torch.testing.assert_close(modified_cache["k"], stock_cache["k"], atol=0, rtol=0)
                    torch.testing.assert_close(modified_cache["v"], stock_cache["v"], atol=0, rtol=0)
                if method.startswith("retrieval"):
                    native_before = {name: value.clone() for name, value in modified_cache.items()}
                    bank_before = {index: tuple(id(tensor) for tensor in pair)
                                   for index, pair in memory.banks[0].items()}
                    read_bytes = sum(t.numel() * t.element_size()
                                     for pair in memory.cached.values() for t in pair)
                    assert memory.release_read_cache() == read_bytes
                    assert not memory.cached
                    assert bank_before == {index: tuple(id(tensor) for tensor in pair)
                                           for index, pair in memory.banks[0].items()}
                    for name, value in modified_cache.items():
                        torch.testing.assert_close(value, native_before[name], atol=0, rtol=0)
        assert memory.retrieval_calls > 0
        if method == "retrieval_reader":
            assert memory.metadata()["learned_reader_layers"] == [0]
        if method in {"retrieval", "retrieval_native", "sink_rephase"}:
            assert memory.trace[-1]["attention_replaced"]
        if "eight" in method:
            assert max(len(row["retrieved_frame_ids"]) for row in memory.trace) > 4
        if "context22" in method:
            changed = [row for row in memory.trace if row["attention_replaced"]]
            assert all(row["retained_tail_latent_frames"] >= 8 for row in changed)
            for row in changed:
                # Current4 plus retained recent history cannot duplicate archive reads.
                first_tail = row["chunk"] * 4 + 4 - row["retained_tail_latent_frames"]
                assert all(index < first_tail for index in row["retrieved_frame_ids"])
        if method.startswith("retrieval"):
            assert all(row.get("retained_tail_latent_frames", 8) >= 4 for row in memory.trace)
    finally:
        memory.close()
