import pytest
import torch

from worldsync.lingbot_local_memory import PackedArchiveTensor, OneShotTextEncoder, archive_storage_bytes, memory_budget


def test_int8_archive_bounds_error_and_counts_fp32_scale_overhead():
    torch.manual_seed(73)
    original = torch.randn(1, 128, 4, 128).to(torch.bfloat16)
    original[:, 0] = 0
    packed = PackedArchiveTensor.pack(original)
    restored = packed.dequantize("cpu", torch.float32)
    assert packed.codes.dtype == torch.int8 and packed.scales.dtype == torch.float32
    assert torch.all((original.float() - restored).abs() <= packed.scales / 2 + 1e-6)
    assert torch.count_nonzero(restored[:, 0]) == 0
    before = original.numel() * original.element_size()
    assert packed.storage_bytes == archive_storage_bytes(before)
    assert packed.storage_bytes / before == .515625
    assert ((restored - original.float()).norm() / original.float().norm()).item() < .01


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_archive_rejects_nonfinite_values(bad):
    with pytest.raises(ValueError, match="non-finite"):
        PackedArchiveTensor.pack(torch.tensor([[bad]]))


def test_one_shot_encoder_preserves_output_then_drops_parameter_payload():
    model = torch.nn.Linear(8, 4)
    class Encoder:
        def __init__(self):
            self.model = model
        def __call__(self, value):
            return self.model(value)
    wrapped = OneShotTextEncoder(Encoder())
    x = torch.randn(2, 8)
    expected = model(x).detach()
    actual = wrapped(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not actual.requires_grad
    assert all(parameter.is_meta for parameter in model.parameters())
    assert wrapped.released_parameter_bytes == (8 * 4 + 4) * 4
    assert wrapped.model.cpu() is model
    with pytest.raises(RuntimeError, match="One-shot"):
        wrapped(x)


def test_memory_budget_keeps_a_real_system_reserve():
    budget = memory_budget(108_000_000_000, 73_000_000_000, 19_765_657_600)
    assert budget["compressed_archive_bytes"] == 10_191_667_200
    assert not budget["conservative_budget_fits"]  # Allocator headroom and16GiB reserve matter.
    assert memory_budget(120_000_000_000, 73_000_000_000, 19_765_657_600)["conservative_budget_fits"]


def test_native_attention_quantizes_only_the_archive_and_preserves_native_kv(monkeypatch):
    import copy
    from pathlib import Path
    import sys
    from types import SimpleNamespace
    from worldsync.lingbot_controls import CameraPlan
    from worldsync.lingbot_paths import PATHS
    from worldsync.lingbot_retrieval import PoseKVMemory
    from worldsync.lingbot_local_memory import Int8ArchiveMemory
    from worldsync.matrix_attention import sdpa_attention
    source = Path(__file__).resolve().parents[1] / "external/LingBot-World2"
    if not source.exists():
        pytest.skip("pinned native checkout not installed")
    existing = sys.modules.get("wan")
    if existing and not Path(existing.__file__).resolve().is_relative_to(source.resolve()):
        import subprocess
        subprocess.run([sys.executable, "-m", "pytest", "-q", f"{__file__}::{sys._getframe().f_code.co_name}"], check=True)
        return
    monkeypatch.syspath_prepend(str(source))
    from wan.modules import model_fast
    monkeypatch.setattr(model_fast, "attention", sdpa_attention)
    torch.manual_seed(17)
    original = model_fast.CausalWanSelfAttention(64, 4, local_attn_size=18, sink_size=6).eval()
    inputs = [torch.randn(1, 16, 64) for _ in PATHS["turn_wait_return"]]
    frequency = torch.arange(1024)[:, None] * torch.linspace(.001, .1, 8)[None]
    freqs = torch.polar(torch.ones_like(frequency), frequency).to(torch.complex128)
    results = []
    for policy in (PoseKVMemory, Int8ArchiveMemory):
        candidate = copy.deepcopy(original)
        pipeline = SimpleNamespace(model=SimpleNamespace(blocks=[SimpleNamespace(self_attn=candidate)]))
        memory = policy(pipeline)
        camera = CameraPlan()
        cache = {"k": torch.zeros(1, 72, 4, 16), "v": torch.zeros(1, 72, 4, 16),
                 "global_end_index": torch.tensor([0]), "local_end_index": torch.tensor([0])}
        observations = []
        try:
            with torch.no_grad():
                for chunk, action in enumerate(PATHS["turn_wait_return"]):
                    camera.advance(action)
                    memory.begin(chunk, camera.poses)
                    memory.clean = True
                    value = candidate(inputs[chunk], torch.tensor([16]), torch.tensor([[4, 2, 2]]),
                                      freqs, cache, chunk * 16, 72, frame_seqlen=4)
                    observations.append((value.clone(), cache["k"].clone(), cache["v"].clone(),
                                         memory.trace[-1]["attention_replaced"]))
                    memory.release_read_cache()
            results.append((observations, memory.metadata()))
        finally:
            memory.close()
    errors = []
    for reference, compressed in zip(results[0][0], results[1][0]):
        torch.testing.assert_close(reference[1], compressed[1], atol=0, rtol=0)
        torch.testing.assert_close(reference[2], compressed[2], atol=0, rtol=0)
        if not reference[3]:
            torch.testing.assert_close(reference[0], compressed[0], atol=0, rtol=0)
        else:
            errors.append(float((reference[0] - compressed[0]).norm() / reference[0].norm()))
    assert errors and 0 < max(errors) < .03
    assert results[1][1]["archive_bytes"] < results[0][1]["archive_bytes"]
    assert results[1][1]["archive_quantization"]["quality_validated"] is False
