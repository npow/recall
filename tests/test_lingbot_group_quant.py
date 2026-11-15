import copy
import sys
from pathlib import Path

import pytest
import torch
from test_lingbot_commit_forcing import native_module
from test_lingbot_nanokv import pipeline

from worldsync.lingbot_group_quant import (
    GroupQuantKV,
    decode_group_tensors,
    encode_group_tensor,
)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(original)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_roundtrip_all_channels_error_storage_and_zeros(bits, dtype):
    torch.manual_seed(6)
    value = torch.randn(1, 9, 3, 128).to(dtype)
    value[:, :1] = 0
    encoded = encode_group_tensor(value, bits=bits)
    decoded = decode_group_tensors([encoded], dtype=dtype, device="cpu")
    assert decoded.shape == value.shape and decoded.dtype == dtype
    assert torch.equal(decoded[:, :1], value[:, :1])
    magnitude = value.float().reshape(1, 9, 3, 4, 32).abs().amax(-1, keepdim=True)
    error = (decoded.float() - value.float()).abs().reshape(1, 9, 3, 4, 32)
    assert torch.all(error <= magnitude * (0.09 if bits == 4 else 0.012))
    stored = (
        encoded["codes"].numel() * encoded["codes"].element_size()
        + encoded["scales"].numel() * 2
    )
    assert stored == value.numel() * (0.5 if bits == 4 else 1) + value.numel() / 32 * 2
    assert encoded["scales"].dtype == torch.bfloat16
    before = decoded.clone()
    value.zero_()
    assert torch.equal(
        before, decode_group_tensors([encoded], dtype=dtype, device="cpu")
    )


@pytest.mark.parametrize("bits", [4, 8])
def test_extremes_are_finite_and_segment_vectorization_matches(bits):
    maximum = torch.finfo(torch.bfloat16).max
    value = (
        torch.tensor(
            [0, maximum, -maximum, torch.finfo(torch.bfloat16).smallest_normal]
        )
        .repeat(128)
        .reshape(1, 4, 1, 128)
    )
    encoded = encode_group_tensor(value.to(torch.bfloat16), bits=bits)
    decoded = decode_group_tensors([encoded], dtype=torch.bfloat16, device="cpu")
    assert torch.isfinite(decoded).all()
    parts = [
        encode_group_tensor(value[:, :1], bits=bits),
        encode_group_tensor(value[:, 1:], bits=bits),
    ]
    combined = decode_group_tensors(parts, dtype=torch.float32, device="cpu")
    separate = torch.cat(
        [
            decode_group_tensors([part], dtype=torch.float32, device="cpu")
            for part in parts
        ],
        dim=1,
    )
    assert torch.equal(combined, separate)


def test_malformed_inputs_and_nonfinite_scales_rejected():
    for kwargs in ({"bits": 3}, {"bits": True}, {"group_size": 3}):
        with pytest.raises(ValueError):
            encode_group_tensor(torch.zeros(1, 1, 1, 128), **kwargs)
    for value in (
        torch.zeros(1, 1, 1, 130),
        torch.tensor([[[[float("nan")] * 128]]]),
        torch.tensor([[[[float("inf")] * 128]]]),
    ):
        with pytest.raises(ValueError):
            encode_group_tensor(value)
    valid = encode_group_tensor(torch.ones(1, 1, 1, 128))
    for broken in (
        {**valid, "bits": 8},
        {**valid, "shape": (1, 1, 1, 127)},
        {**valid, "codes": valid["codes"].float()},
        {**valid, "scales": valid["scales"].float()},
        {**valid, "scales": torch.zeros_like(valid["scales"])},
        {**valid, "scales": torch.full_like(valid["scales"], float("nan"))},
        None,
    ):
        with pytest.raises(ValueError):
            decode_group_tensors([broken], dtype=torch.bfloat16, device="cpu")
    with pytest.raises(ValueError):
        decode_group_tensors([], dtype=torch.bfloat16, device="cpu")


def fixture_model(native, dtype):
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = native.CausalWanSelfAttention(
                512, 4, local_attn_size=18, sink_size=6
            )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block(), Block()])

        def forward(self, x, t, caches, start, freqs):
            for block, cache in zip(self.blocks, caches):
                x = x + block.self_attn(
                    x,
                    torch.tensor([16]),
                    torch.tensor([[4, 2, 2]]),
                    freqs,
                    cache,
                    start,
                    72,
                    frame_seqlen=4,
                )
            return x

    torch.manual_seed(41)
    model = Model().eval().to(dtype)
    phase = torch.arange(1024)[:, None] * torch.linspace(0.001, 0.1, 64)[None]
    frequencies = torch.polar(torch.ones_like(phase), phase).to(torch.complex128)
    return model, frequencies


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_native_first_block_exact_later_bounded_causal_rolling_cleanup(
    monkeypatch, request, bits, dtype
):
    native = native_module(monkeypatch, request)
    if native is None:
        return
    model, freqs = fixture_model(native, dtype)
    stock = copy.deepcopy(model)
    stock_pipe, quantized_pipe = pipeline(stock), pipeline(model)
    stock_cache = stock_pipe._initialize_self_kv_cache(2, [1, 72, 4, 128], dtype, "cpu")
    initial_versions = {name: p._version for name, p in model.named_parameters()}
    initial_methods = [block.self_attn.forward.__func__ for block in model.blocks]
    adapter = GroupQuantKV(
        quantized_pipe,
        Path(__file__).resolve().parents[1] / "external/NanoKV",
        capacity_frames=18,
        bits=bits,
    )
    errors = []
    with torch.no_grad(), adapter:
        caches = quantized_pipe._initialize_self_kv_cache(
            2, [1, 72, 4, 128], dtype, "cpu"
        )
        for chunk in range(8):
            protected = [
                [
                    (part[target]["codes"].clone(), part[target]["scales"].clone())
                    for part in cache["codec_segments"]
                    for target in ("k", "v")
                ]
                for cache in caches
            ]
            for timestep in (1000, 500, 0):
                x = torch.randn(1, 16, 512).to(dtype)
                expected = stock(
                    x, torch.tensor([timestep]), stock_cache, chunk * 16, freqs
                )
                actual = model(x, torch.tensor([timestep]), caches, chunk * 16, freqs)
                if chunk == 0:
                    assert torch.equal(expected, actual)
                error = (
                    actual.float() - expected.float()
                ).square().mean().sqrt() / expected.float().square().mean().sqrt()
                errors.append(float(error))
                assert torch.isfinite(actual).all() and error < (
                    0.10 if bits == 4 else 0.03
                )
                for original, quantized in zip(stock_cache, caches):
                    assert torch.equal(
                        original["global_end_index"], quantized["global_end_index"]
                    )
                    assert torch.equal(
                        original["local_end_index"], quantized["local_end_index"]
                    )
                if timestep:
                    for before, cache in zip(protected, caches):
                        after = [
                            (part[target]["codes"], part[target]["scales"])
                            for part in cache["codec_segments"]
                            for target in ("k", "v")
                        ]
                        assert len(before) == len(after)
                        assert all(
                            torch.equal(a, c) and torch.equal(b, d)
                            for (a, b), (c, d) in zip(before, after)
                        )
        assert max(errors) > 0
        assert adapter.operator_bytes == 0 and adapter.payload_peak_bytes > 0
    meta = adapter.metadata()
    assert (
        meta["identity"] is False
        and meta["codec_sha256"] is None
        and not meta["fitted_operators"]
    )
    assert meta["group_bits"] == bits and meta["group_size"] == 32 and meta["closed"]
    assert initial_versions == {
        name: p._version for name, p in model.named_parameters()
    }
    assert initial_methods == [
        block.self_attn.forward.__func__ for block in model.blocks
    ]
    assert not model._forward_hooks and not model._forward_pre_hooks
    assert not any(name.startswith(adapter.namespace) for name in sys.modules)


@pytest.mark.parametrize("bits", [4, 8])
def test_clean_encoding_failure_restores_hooks_and_bindings(monkeypatch, request, bits):
    native = native_module(monkeypatch, request)
    if native is None:
        return
    model, freqs = fixture_model(native, torch.bfloat16)
    pipe = pipeline(model)
    initial = [block.self_attn.forward.__func__ for block in model.blocks]
    adapter = GroupQuantKV(
        pipe,
        Path(__file__).resolve().parents[1] / "external/NanoKV",
        capacity_frames=18,
        bits=bits,
    )
    with pytest.raises(ValueError, match="nonfinite KV"), torch.no_grad(), adapter:
        cache = pipe._initialize_self_kv_cache(
            2, [1, 72, 4, 128], torch.bfloat16, "cpu"
        )
        model(
            torch.full((1, 16, 512), float("nan"), dtype=torch.bfloat16),
            torch.tensor([0]),
            cache,
            0,
            freqs,
        )
    assert adapter.closed and not adapter.clean and not adapter.caches
    assert [block.self_attn.forward.__func__ for block in model.blocks] == initial
    assert not model._forward_hooks and not model._forward_pre_hooks
    assert "_initialize_self_kv_cache" not in pipe.__dict__
    assert not any(name.startswith(adapter.namespace) for name in sys.modules)
