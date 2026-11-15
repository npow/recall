"""Long logical positions/rolling boundaries using actual native attention."""

import copy
import types
from pathlib import Path

import pytest
import torch
from test_lingbot_commit_forcing import fixture_model, native_module
from test_lingbot_nanokv import pipeline

from worldsync.lingbot_nanokv import NativeNanoKV


@pytest.fixture(autouse=True)
def one_cpu_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(original)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_identity130_matches_native_after_140_latent_frames(
    monkeypatch, request, dtype
):
    native = native_module(monkeypatch, request)
    if native is None:
        return
    model, freqs, _ = fixture_model(native, dtype)
    for block in model.blocks:
        block.self_attn.local_attn_size = 130

    def forward(self, x, t, caches, start, freqs):
        for block, cache in zip(self.blocks, caches):
            x = x + block.self_attn(
                x,
                torch.tensor([16]),
                torch.tensor([[4, 2, 2]]),
                freqs,
                cache,
                start,
                520,
                frame_seqlen=4,
            )
        return x

    model.forward = types.MethodType(forward, model)
    control = copy.deepcopy(model)
    stock_pipe, compressed_pipe = pipeline(control), pipeline(model)
    control_cache = stock_pipe._initialize_self_kv_cache(
        2, [1, 520, 4, 16], dtype, "cpu"
    )
    adapter = NativeNanoKV(
        compressed_pipe,
        source_root=Path(__file__).resolve().parents[1] / "external/NanoKV",
        capacity_frames=130,
        identity=True,
    )
    with torch.no_grad(), adapter:
        cache = compressed_pipe._initialize_self_kv_cache(
            2, [1, 520, 4, 16], dtype, "cpu"
        )
        for chunk in range(35):
            for timestep in (1000, 500, 0):
                latent = torch.randn(1, 16, 64).to(dtype)
                expected = control(
                    latent, torch.tensor([timestep]), control_cache, chunk * 16, freqs
                )
                actual = model(
                    latent, torch.tensor([timestep]), cache, chunk * 16, freqs
                )
                assert torch.equal(expected, actual), (chunk, timestep)
                for baseline, identity in zip(control_cache, cache):
                    assert torch.equal(
                        baseline["global_end_index"], identity["global_end_index"]
                    )
                    assert torch.equal(
                        baseline["local_end_index"], identity["local_end_index"]
                    )
        assert max(row["current_start"] for row in adapter.trace) == 136 * 4
        assert all(row["read_key_tokens"] == 520 for row in adapter.trace[-18:])
        assert all(int(item["global_end_index"].item()) == 140 * 4 for item in cache)
    assert adapter.closed and not model._forward_hooks and not model._forward_pre_hooks
