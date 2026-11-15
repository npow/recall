import torch

from worldsync.lingbot_fast_metadata import HostCacheIndex


def test_host_cache_index_tracks_rollover_and_preserves_native_tensor_writes():
    tensor = torch.zeros(1, dtype=torch.long)
    mirrored = HostCacheIndex(tensor)
    for position in (0, 6032, 12064, 27144, 24128, 27144):
        mirrored.fill_(position)
        assert mirrored.item() == tensor.item() == position
