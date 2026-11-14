"""Exercise the real upstream attention on small tensors without checkpoints."""
import copy
from pathlib import Path
import sys

import pytest
import torch

from worldsync.memory import MemoryConfig


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["window", "landmark"])
def test_adapter_attention(device, mode):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    repo = Path(__file__).resolve().parents[1] / "external/ABot-World"
    if not repo.is_dir():
        pytest.skip("run scripts/setup.py first")
    sys.path.insert(0, str(repo))
    import wan.modules.causal_model as causal
    causal._REL_ROPE_CACHE_ENABLED = False
    # Same SDPA path, including on CUDA; keep the backend independent of local extras.
    import importlib
    attn_module = importlib.import_module("wan.modules.attention")
    attn_module.DEFAULT_ATTN_BACKEND = "sdpa"
    original = causal.CausalWanSelfAttention(24, 2, local_attn_size=21, use_relative_rope=True).to(device)
    adapted = copy.deepcopy(original)
    adapted._worldsync_config = MemoryConfig(mode=mode)
    freqs = torch.cat([causal.rope_params(1024, 4)] * 3, dim=1).to(device)
    grid = torch.tensor([[3, 2, 2]], device=device)
    lengths = torch.tensor([12], device=device)

    def cache():
        return {"k": torch.zeros(1, 21 * 4, 2, 12, device=device),
                "v": torch.zeros(1, 21 * 4, 2, 12, device=device),
                "global_end_index": torch.tensor([0], device=device),
                "local_end_index": torch.tensor([0], device=device)}

    baseline_cache, adapted_cache = cache(), cache()
    with torch.no_grad():
        for frame in range(0, 33, 3):
            for _ in range(2):  # provisional denoising then clean-context overwrite
                x = torch.randn(1, 12, 24, device=device)
                a = original(x, lengths, grid, freqs, kv_cache=baseline_cache, current_start=frame * 4)
                b = adapted(x, lengths, grid, freqs, kv_cache=adapted_cache, current_start=frame * 4)
                assert b.shape == a.shape and torch.isfinite(b).all()
                if mode == "window" or frame < 21:
                    torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
                bank = adapted_cache["_worldsync"]
                assert len(bank.landmarks + bank.recent + bank.pending) <= 21
