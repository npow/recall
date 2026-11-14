import pytest
import torch

from worldsync.memory import LandmarkMemory, MemoryConfig, memory_view


def kv(value, frames=3):
    # Positive/negative vectors create distinct scene-entry descriptors.
    key = torch.full((1, frames * 4, 2, 12), float(value))
    return key, key.clone()


def test_denoising_does_not_commit_noisy_copies():
    memory = LandmarkMemory(MemoryConfig())
    for value in (1, 2, 3, 4):
        memory.view(*kv(value), 0, 4)
    k, _ = memory.view(*kv(5), 3, 4)
    assert memory.committed_frames == 3
    assert [f.index for f in memory.recent] == [0, 1, 2]
    torch.testing.assert_close(k[:, :12], kv(4)[0])


def test_window_matches_fifo_after_many_evictions():
    memory = LandmarkMemory(MemoryConfig(mode="window"))
    for frame in range(0, 90, 3):
        key, _ = memory.view(*kv(frame + 1), frame, 4)
        assert key.shape[1] <= 21 * 4
    assert [f.index for f in memory.recent + memory.pending] == list(range(90 - 21, 90))


def test_distinct_old_view_survives_with_bounded_memory():
    memory = LandmarkMemory(MemoryConfig(novelty_threshold=0.5))
    for start in range(0, 300, 3):
        k, _ = memory.view(*kv(1 if start == 0 else -1), start, 4)
        assert k.shape[1] <= 21 * 4
    assert 0 in [f.index for f in memory.landmarks]
    assert len(memory.recent + memory.pending) == 15
    assert len(memory.landmarks) <= 6


def test_reused_cache_resets_between_episodes():
    class Module:
        _worldsync_config = MemoryConfig()
    cache = {}
    module = Module()
    memory_view(module, cache, *kv(1), 0, 4)
    memory_view(module, cache, *kv(2), 12, 4)
    memory_view(module, cache, *kv(9), 0, 4)
    assert cache["_worldsync"].committed_frames == 0
    assert cache["_worldsync"].pending_start == 0


def test_reject_discontinuous_or_invalid_streams():
    memory = LandmarkMemory(MemoryConfig())
    memory.view(*kv(1), 0, 4)
    with pytest.raises(ValueError, match="non-contiguous"):
        memory.view(*kv(2), 6, 4)
    with pytest.raises(ValueError):
        MemoryConfig(window_frames=22)
