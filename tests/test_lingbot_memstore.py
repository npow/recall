import pytest

torch = pytest.importorskip("torch")
from worldsync import lingbot_memstore as ms

HW = (6, 8)


def kv(seed=0, tokens=48, heads=2, dim=16):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, tokens, heads, dim, generator=g), torch.randn(1, tokens, heads, dim, generator=g)


def relative(a, b):
    return float((a - b).norm() / b.norm())


def test_exact_roundtrip_is_identical():
    k, v = kv()
    codec = ms.make_codec("exact")
    dk, dv, index = codec.decode(codec.encode(k, v, HW), k.dtype)
    assert index is None and torch.equal(dk, k) and torch.equal(dv, v)


@pytest.mark.parametrize("name,bound", [("fp8", 0.06), ("int8", 0.03)])
def test_eight_bit_error_and_size(name, bound):
    k, v = kv()
    codec = ms.make_codec(name)
    stored = codec.encode(k.to(torch.bfloat16), v.to(torch.bfloat16), HW)
    dk, dv, _ = codec.decode(stored, torch.float32)
    assert relative(dk, k) < bound and relative(dv, v) < bound
    exact = ms.nbytes(ms.make_codec("exact").encode(k.to(torch.bfloat16), v.to(torch.bfloat16), HW))
    assert ms.nbytes(stored) < 0.6 * exact  # about half of bf16, not a quarter


def test_pool_keeps_token_count_and_block_means():
    k, v = kv()
    codec = ms.make_codec("pool2")
    stored = codec.encode(k, v, HW)
    dk, dv, index = codec.decode(stored, k.dtype)
    assert index is None and dk.shape == k.shape
    block = k.view(1, 6, 8, 2, 16)[:, :2, :2].mean(dim=(1, 2))
    assert torch.allclose(dk.view(1, 6, 8, 2, 16)[:, 0, 0], block, atol=1e-5)
    assert torch.allclose(dk.view(1, 6, 8, 2, 16)[:, 1, 1], block, atol=1e-5)
    constant = torch.ones_like(k)
    assert torch.allclose(codec.decode(codec.encode(constant, constant, HW), k.dtype)[0], constant)
    assert ms.nbytes(stored) * 3.9 < ms.nbytes(ms.make_codec("exact").encode(k, v, HW)) + 1


def test_pool4_handles_grid_not_divisible():
    k, v = kv()
    codec = ms.make_codec("pool4")
    dk, _, _ = codec.decode(codec.encode(k, v, HW), k.dtype)
    assert dk.shape == k.shape  # 6x8 -> 2x2 blocks (ceil), cropped back


def test_top_tokens_keep_largest_values_and_indices():
    k, v = kv()
    v[:, 5] *= 100
    codec = ms.make_codec("top25")
    stored = codec.encode(k, v, HW)
    dk, dv, index = codec.decode(stored, k.dtype)
    assert len(index) == 12 and 5 in index.tolist() and torch.equal(dk, k[:, index])
    assert ms.make_codec("rand25").encode(k, v, HW)["index"].tolist() == ms.make_codec("rand25").encode(v, k, HW)["index"].tolist()


def test_low_rank_recovers_a_low_rank_matrix():
    g = torch.Generator().manual_seed(1)
    left, right = torch.randn(48, 4, generator=g), torch.randn(4, 32, generator=g)
    k = (left @ right).view(1, 48, 2, 16)
    codec = ms.make_codec("lr8")
    dk, dv, _ = codec.decode(codec.encode(k, k, HW), torch.float32)
    assert relative(dk, k) < 2e-2


def test_composition_and_layer_sets():
    k, v = kv()
    codec = ms.make_codec("pool2+fp8")
    stored = codec.encode(k.to(torch.bfloat16), v.to(torch.bfloat16), HW)
    assert codec.decode(stored, torch.float32)[0].shape == k.shape
    assert ms.nbytes(stored) * 6 < ms.nbytes(ms.make_codec("exact").encode(k.to(torch.bfloat16), v.to(torch.bfloat16), HW))
    assert ms.layer_set("even", 40) == list(range(0, 40, 2)) and len(ms.layer_set("every4", 40)) == 10
    assert ms.layer_set("first10", 40) == list(range(10)) and ms.layer_set("last10", 40) == list(range(30, 40))
    assert ms.layer_set("mid20", 40) == list(range(10, 30)) and len(ms.layer_set("last20", 40)) == 20
    with pytest.raises(ValueError):
        ms.make_codec("zip")


def test_positioned_keys_match_full_frame_rope_on_the_kept_tokens():
    k, _ = kv()

    def rope(x, grid, freqs, start_frame=0):  # stand-in: position-dependent, token-wise
        scale = torch.arange(x.shape[1], dtype=x.dtype).view(1, -1, 1, 1) + 10 * start_frame + 1
        return x * scale

    grid = torch.tensor([[3, 6, 8]])
    full = ms.positioned_keys(rope, k, None, 48, 7, grid, None)
    index = torch.tensor([0, 5, 47])
    part = ms.positioned_keys(rope, k[:, index], index, 48, 7, grid, None)
    assert torch.equal(part, full[:, index])


def _fake_memories(monkeypatch, codec_name, layers=None, rope="native"):
    """Drive BudgetPoseMemory and StoredBudgetMemory through the same archive + read on a fake model."""
    import sys
    import types
    import numpy as np
    from worldsync.lingbot_controls import CameraPlan

    seen = []
    fake = types.ModuleType("wan.modules.model_fast")
    fake.attention = lambda q, k, v, **kw: seen.append((k.clone(), v.clone())) or q

    def rope_apply(x, grid, freqs, start_frame=0):
        f, h, w = grid.tolist()[0]
        phase = (torch.arange(f * h * w, dtype=torch.float32).view(1, -1, 1, 1) * 0.01 + start_frame)
        return (x.float() * torch.cos(phase) + x.float().flip(-1) * torch.sin(phase)).type_as(x)

    fake.causal_rope_apply = rope_apply
    package = types.ModuleType("wan")
    modules = types.ModuleType("wan.modules")
    modules.model_fast = fake
    package.modules = modules
    for name, module in (("wan", package), ("wan.modules", modules), ("wan.modules.model_fast", fake)):
        monkeypatch.setitem(sys.modules, name, module)

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm_k = torch.nn.Identity()

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()

    pipeline = types.SimpleNamespace(model=types.SimpleNamespace(blocks=[Block(), Block(), Block(), Block()]))
    from worldsync.lingbot_budgetmem import BudgetPoseMemory, StoredBudgetMemory
    frame_tokens, heads, dim = 48, 2, 16
    grid, window = torch.tensor([[4, 6, 8]]), 18
    camera = CameraPlan(turn_degrees_per_second=68)
    actions = ["WAIT"] + ["RIGHT"] * 3 + ["WAIT"] * 4 + ["LEFT"] * 3 + ["WAIT"]
    results = []
    for build in (lambda: BudgetPoseMemory(pipeline, retrieved=4, archive_frames=24, rope=rope),
                  lambda: StoredBudgetMemory(pipeline, ms.make_codec(codec_name), layers=layers, retrieved=4,
                                             archive_frames=24, rope=rope)):
        seen.clear()
        memory = build()
        camera = CameraPlan(turn_degrees_per_second=68)
        generator = torch.Generator().manual_seed(3)
        final = {}
        for chunk, action in enumerate(actions):
            camera.advance(action)
            memory.begin(chunk, camera.poses)
            first = chunk * 4
            for layer in range(4):
                query = torch.randn(1, 4 * frame_tokens, heads, dim, generator=generator)
                key = torch.randn(1, min(first + 4, window) * frame_tokens, heads, dim, generator=generator)
                value = torch.randn(key.shape, generator=generator)
                for clean in (False, True):
                    memory.clean = clean
                    memory.active = (layer, first * frame_tokens, frame_tokens, grid, None)
                    memory.raw_key = key[:, -query.shape[1]:].clone()
                    seen.clear()
                    memory.attend(query, key, value)
                    final[(chunk, layer, clean)] = seen[-1]
                memory.active = None
        results.append((final, memory.metadata(), [t["retrieved_frame_ids"] for t in memory.trace]))
        memory.close()
    return results


def test_exact_codec_reads_match_the_reference_memory(monkeypatch):
    (reference, _, picked), (stored, meta, picked_stored) = _fake_memories(monkeypatch, "exact")
    assert picked == picked_stored and any(picked)
    assert reference.keys() == stored.keys()
    for name in reference:
        assert torch.allclose(reference[name][0], stored[name][0], atol=1e-6), name
        assert torch.equal(reference[name][1], stored[name][1]), name
    assert meta["storage_codec"] == "exact" and meta["stored_bytes_per_archived_frame"] > 0


def test_layer_subset_leaves_other_layers_on_the_stock_window(monkeypatch):
    (reference, _, _), (stored, meta, _) = _fake_memories(monkeypatch, "exact", layers=[1, 3])
    last = max(chunk for chunk, _, _ in reference)
    assert stored[(last, 0, False)][0].shape[1] == 18 * 48          # layer 0: stock window untouched
    assert stored[(last, 1, False)][0].shape[1] == (6 + 4 + 8) * 48  # layer 1: sink + retrieved + tail
    assert torch.allclose(reference[(last, 1, False)][0], stored[(last, 1, False)][0], atol=1e-6)
    assert meta["storage_layers"] == [1, 3]


def test_token_subset_shrinks_the_attended_memory(monkeypatch):
    (_, _, _), (stored, meta, _) = _fake_memories(monkeypatch, "top25")
    last = max(chunk for chunk, _, _ in stored)
    assert stored[(last, 1, False)][0].shape[1] == (6 + 8) * 48 + 4 * 12
    assert 3.5 < meta["compression_ratio"] * (2 * 48 * 32 * 4 * 4) / (ms.EXACT_BYTES_PER_LAYER * 4) < 4.5


def test_low_rank_accepts_bf16_under_autocast():
    k, v = kv()
    codec = ms.make_codec("lr8")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        stored = codec.encode(k.to(torch.bfloat16), v.to(torch.bfloat16), HW)
        dk, _, _ = codec.decode(stored, torch.bfloat16)
    assert stored["kl"].dtype == torch.float16 and dk.dtype == torch.bfloat16 and dk.shape == k.shape
