import copy
import sys
from pathlib import Path

import pytest
import torch
from test_lingbot_commit_forcing import fixture_model, native_module

from worldsync.lingbot_nanokv import NativeNanoKV, validate_codec

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_cpu_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(original)


def pipeline(model):
    class Pipeline:
        def __init__(self):
            self.model = model

        def _initialize_self_kv_cache(self, num_layers, shape, dtype, device):
            return [
                {
                    "k": torch.zeros(shape, dtype=dtype, device=device),
                    "v": torch.zeros(shape, dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], device=device),
                    "local_end_index": torch.tensor([0], device=device),
                }
                for _ in range(num_layers)
            ]

    return Pipeline()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_identity_matches_every_noisy_clean_forward_and_rolling_cache(
    monkeypatch, request, dtype
):
    native = native_module(monkeypatch, request)
    if native is None:
        return
    model, freqs, make_cache = fixture_model(native, dtype)
    control = copy.deepcopy(model)
    pipe = pipeline(model)
    control_cache = make_cache()
    versions = {name: p._version for name, p in model.named_parameters()}
    original = [block.self_attn.forward.__func__ for block in model.blocks]
    adapter = NativeNanoKV(
        pipe, source_root=ROOT / "external/NanoKV", capacity_frames=18, identity=True
    )
    with torch.no_grad(), adapter:
        cache = pipe._initialize_self_kv_cache(2, [1, 72, 4, 16], dtype, "cpu")
        assert all(item["k"].shape[1] == 1 for item in cache)
        for chunk in range(9):
            # Four independent noisy evaluations followed by the actual clean commit.
            for timestep in (1000, 750, 500, 250, 0):
                x = torch.randn(1, 16, 64).to(dtype)
                expected = control(
                    x, torch.tensor([timestep]), control_cache, chunk * 16, freqs
                )
                actual = model(x, torch.tensor([timestep]), cache, chunk * 16, freqs)
                assert torch.equal(expected, actual), (chunk, timestep)
                for stock, compressed in zip(control_cache, cache):
                    assert torch.equal(
                        stock["global_end_index"], compressed["global_end_index"]
                    )
                    assert torch.equal(
                        stock["local_end_index"], compressed["local_end_index"]
                    )
                if timestep == 0:
                    # Compare every reconstructed clean cache byte, including sinks
                    # and a rolling boundary cutting through a stored chunk.
                    for stock, compressed in zip(control_cache, cache):
                        history = adapter.runtime._decode_active_history(
                            compressed["codec_segments"],
                            committed_end=compressed["codec_committed_end"],
                            capacity=72,
                            sink_tokens=24,
                            dtype=dtype,
                            device="cpu",
                        )
                        key = adapter.runtime.rope_pre_key_by_positions(
                            history[0],
                            history[2],
                            grid_sizes=torch.tensor([[4, 2, 2]]),
                            freqs=freqs,
                            frame_seqlen=4,
                        )
                        length = int(stock["local_end_index"].item())
                        assert torch.equal(key, stock["k"][:, :length])
                        assert torch.equal(history[1], stock["v"][:, :length])
        assert all(not item["codec_commit"] for item in cache)
        assert len(adapter.trace) == 90
        assert adapter.payload_peak_bytes > 0
    assert adapter.closed
    assert all(not item["codec_segments"] for item in cache)
    assert "_initialize_self_kv_cache" not in pipe.__dict__
    assert [block.self_attn.forward.__func__ for block in model.blocks] == original
    assert {name: p._version for name, p in model.named_parameters()} == versions
    assert not any(name.startswith(adapter.namespace) for name in sys.modules)
    assert not model._forward_hooks and not model._forward_pre_hooks


def test_exception_cleans_phase_and_restores_existing_allocator(monkeypatch, request):
    native = native_module(monkeypatch, request)
    if native is None:
        return
    model, freqs, _ = fixture_model(native, torch.float32)
    pipe = pipeline(model)
    existing = pipe._initialize_self_kv_cache
    pipe._initialize_self_kv_cache = existing
    adapter = NativeNanoKV(
        pipe, source_root=ROOT / "external/NanoKV", capacity_frames=18, identity=True
    )
    with pytest.raises(ValueError, match="logical cache"), torch.no_grad(), adapter:
        caches = pipe._initialize_self_kv_cache(2, [1, 71, 4, 16], torch.float32, "cpu")
        model(torch.randn(1, 16, 64), torch.tensor([0]), caches, 0, freqs)
    assert adapter.closed and not adapter.clean
    assert pipe._initialize_self_kv_cache is existing
    assert not model._forward_hooks and not model._forward_pre_hooks
    adapter.close()


def test_partial_attachment_failure_restores_everything(monkeypatch, request):
    native = native_module(monkeypatch, request)
    if native is None:
        return
    model, _, _ = fixture_model(native, torch.float32)
    pipe = pipeline(model)
    model.blocks[1].self_attn.memory_reader = torch.nn.Identity()
    adapter = NativeNanoKV(
        pipe, source_root=ROOT / "external/NanoKV", capacity_frames=18, identity=True
    )
    with pytest.raises(ValueError, match="readers/retrieval"):
        adapter.__enter__()
    assert "forward" not in model.blocks[0].self_attn.__dict__
    assert "_initialize_self_kv_cache" not in pipe.__dict__
    assert not any(name.startswith(adapter.namespace) for name in sys.modules)


def test_rejects_unverified_artifact_before_deserialization(tmp_path, monkeypatch):
    fake = tmp_path / "codec.pt"
    fake.write_bytes(b"invalid")

    def forbidden(*args, **kwargs):
        raise AssertionError("Unverified pickle must never be loaded")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="SHA256"):
        validate_codec(fake)


def test_source_drift_rejected_before_attachment(monkeypatch, request, tmp_path):
    native = native_module(monkeypatch, request)
    if native is None:
        return
    model, _, _ = fixture_model(native, torch.float32)
    path = tmp_path / "models/LingBot-World-2.0/compression/wan/modules"
    path.mkdir(parents=True)
    (path / "lingbot_v2_cache_runtime.py").write_text("# changed")
    adapter = NativeNanoKV(
        pipeline(model), source_root=tmp_path, capacity_frames=18, identity=True
    )
    with pytest.raises(ValueError, match="helper changed"):
        adapter.__enter__()
    assert all("forward" not in block.self_attn.__dict__ for block in model.blocks)


def test_released_codec_metadata_on_cpu_when_staged():
    path = ROOT / "weights/nanokv/lingbot-world-2.0/codec.pt"
    if not path.exists():
        pytest.skip("Published codec not staged; no download in unit tests")
    packed = validate_codec(path)
    assert len(packed["entries"]) == 80
    assert all(
        item["encoder"]["qweight"].device.type == "cpu"
        for item in packed["entries"].values()
    )


def test_official_codec_math_cpu_int4_and_clean_only_segments():
    # Tiny deterministic operators exercise the real released encoder/decoder
    # and cache lifecycle; GPU Triton dispatch remains a separate validation.
    adapter = NativeNanoKV(
        pipeline(torch.nn.Linear(1, 1)),
        source_root=ROOT / "external/NanoKV",
        capacity_frames=18,
        identity=True,
    )
    try:
        adapter._helpers()
        torch.manual_seed(42)

        def operator(
            root, *, target, sp_rank, layer, joint_dim, rank, device, linear_root=None
        ):
            return {
                "mean": torch.zeros(16),
                "encoder": torch.eye(16)[:8].to(torch.bfloat16),
                "decoder": torch.eye(16)[:, :8].to(torch.bfloat16),
            }

        adapter.codec._load_operator = operator
        # Restore the official encode/decode methods after identity harness setup.
        adapter.runtime.encode_pair = adapter.codec.encode_pair
        adapter.runtime.decode_segments = adapter.codec.decode_segments
        cache = {
            "k": torch.zeros(1, 1, 1, 16),
            "v": torch.zeros(1, 1, 1, 16),
            "global_end_index": torch.tensor([0]),
            "local_end_index": torch.tensor([0]),
        }
        adapter.runtime.configure_cache(
            cache,
            layer=0,
            sp_rank=0,
            options={
                "codec_rank": 8,
                "codec_clusters": 2,
                "codec_iters": 2,
                "codec_basis_root_k": "synthetic",
                "codec_basis_root_v": "synthetic",
            },
        )
        for chunk in range(7):
            key, value = [torch.randn(1, 4, 1, 16).to(torch.bfloat16) for _ in range(2)]
            for clean in (False, False, True):
                before = len(cache["codec_segments"])
                cache["codec_commit"] = clean
                keys, values, positions = adapter.runtime.codec_cache_attention_inputs(
                    cache,
                    key,
                    value,
                    current_start=chunk * 4,
                    capacity=18,
                    sink_tokens=6,
                    max_attention_size=18,
                )
                assert keys.shape[1] == min(18, (chunk + 1) * 4)
                assert values.shape == keys.shape and torch.isfinite(keys).all()
                assert positions.numel() == keys.shape[1]
                if not clean:
                    assert len(cache["codec_segments"]) == before
                    assert cache["codec_committed_end"] == chunk * 4
            assert cache["codec_committed_end"] == (chunk + 1) * 4
            assert all(
                segment["k"]["latent_packed"].dtype == torch.uint8
                for segment in cache["codec_segments"]
            )
    finally:
        adapter.close()
