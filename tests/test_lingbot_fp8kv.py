import pytest

torch = pytest.importorskip("torch")
from worldsync import lingbot_fp8kv as fp8


def test_quantize_roundtrip_error_is_small_and_scales_per_token_head():
    torch.manual_seed(0)
    x = torch.randn(1, 6, 4, 128, dtype=torch.bfloat16) * torch.tensor([0.1, 1.0, 10.0, 100.0]).view(1, 1, 4, 1).bfloat16()
    q, scale = fp8.quantize(x)
    assert q.dtype == fp8.FP8 and scale.shape == (1, 6, 4, 1)
    back = fp8.dequantize(q, scale)
    rel = (back.float() - x.float()).norm(dim=-1) / x.float().norm(dim=-1)
    assert rel.max() < 0.08  # e4m3 has 3 mantissa bits; per-head scaling keeps small heads accurate


def test_zero_rows_stay_zero_and_nothing_is_nan():
    x = torch.zeros(1, 3, 2, 16, dtype=torch.bfloat16)
    x[0, 1] = 1e4
    q, scale = fp8.quantize(x)
    back = fp8.dequantize(q, scale)
    assert not torch.isnan(back.float()).any()
    assert back[0, 0].abs().max() == 0


def test_store_roll_read_matches_plain_tensor_window():
    torch.manual_seed(1)
    shape = (1, 12, 2, 32)
    cache = fp8.allocate(1, shape, torch.bfloat16, "cpu", audit_layers={0})[0]
    keys = torch.randn(1, 12, 2, 32, dtype=torch.bfloat16)
    vals = torch.randn(1, 12, 2, 32, dtype=torch.bfloat16)
    fp8.store(cache, 0, 12, keys, vals)
    fp8.roll(cache, 4, 8, 8, 12)      # evict tokens 4..7 like the sink+rolling window
    assert torch.equal(cache["k_ref"][:, 4:8], keys[:, 8:12])
    k, v = fp8.read(cache, 0, 8)
    ref_k = torch.cat([keys[:, :4], keys[:, 8:12]], dim=1)
    assert (k.float() - ref_k.float()).norm() / ref_k.float().norm() < 0.08
    assert v.shape == (1, 8, 2, 32)


def test_fp8_cache_uses_about_half_the_bytes():
    shape = (1, 100, 8, 128)
    bf16 = 2 * 2 * 100 * 8 * 128
    entry = fp8.allocate(1, shape, torch.bfloat16, "cpu")[0]
    used = sum(entry[name].numel() * entry[name].element_size() for name in ("k", "v", "ks", "vs"))
    assert used < 0.52 * bf16
