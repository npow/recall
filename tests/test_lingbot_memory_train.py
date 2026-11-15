import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from worldsync import lingbot_lora
from worldsync.lingbot_memory_data import latent_index, pan_profile

SOURCE = Path(__file__).resolve().parents[1] / "external/LingBot-World2"


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(16, 16) for _ in range(4))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn, self.cross_attn = Attention(), Attention()


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])


def test_fresh_adapter_is_identity_and_only_adapters_train():
    model = Tiny()
    x = torch.randn(3, 16)
    before = model.blocks[0].self_attn.q(x)
    params = lingbot_lora.apply_lora(model, rank=4, alpha=8)
    assert torch.allclose(model.blocks[0].self_attn.q(x), before)
    assert len(params) == 2 * 8 * 2  # a and b for q,k,v,o in self and cross attention, two blocks
    assert {n for n, p in model.named_parameters() if p.requires_grad} == {n for n, _ in model.named_parameters() if n.endswith(("lora_a", "lora_b"))}


def test_save_load_roundtrip_restores_outputs(tmp_path):
    import copy
    base = Tiny()
    model, fresh = copy.deepcopy(base), copy.deepcopy(base)
    lingbot_lora.apply_lora(model, rank=4, alpha=8)
    for p in lingbot_lora.trainable_parameters(model):
        p.data.normal_()
    x = torch.randn(3, 16)
    expected = model.blocks[1].cross_attn.o(x)
    path = tmp_path / "lora.pt"
    lingbot_lora.save_lora(model, path)
    meta = lingbot_lora.load_lora(fresh, path)
    assert meta["rank"] == 4
    assert torch.allclose(fresh.blocks[1].cross_attn.o(x), expected, atol=1e-6)


def test_load_rejects_wrong_layers(tmp_path):
    model = Tiny()
    lingbot_lora.apply_lora(model, rank=2, alpha=2)
    path = tmp_path / "lora.pt"
    lingbot_lora.save_lora(model, path)
    bigger = Tiny()
    bigger.blocks.append(Block())
    with pytest.raises(ValueError):
        lingbot_lora.load_lora(bigger, path)


def test_rope_positions_match_upstream_for_contiguous_frames():
    if not SOURCE.exists():
        pytest.skip("LingBot source not checked out")
    sys.path.insert(0, str(SOURCE))
    try:
        from wan.modules.model_fast import causal_rope_apply
        from wan.modules.model import rope_params
    except Exception as error:  # missing optional deps such as diffusers
        pytest.skip(f"upstream import unavailable: {error}")
    from worldsync.lingbot_memory_train import rope_at
    d, f, h, w, start = 128, 3, 4, 5, 7
    freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1)
    x = torch.randn(1, f * h * w, 2, d)
    reference = causal_rope_apply(x, torch.tensor([[f, h, w]]), freqs, start_frame=start)
    ours = rope_at(x, list(range(start, start + f)), h, w, freqs)
    assert torch.allclose(reference, ours, atol=1e-5)
    shifted = rope_at(x, [start, start + 5, start + 6], h, w, freqs)  # non-contiguous virtual slots differ
    assert not torch.allclose(reference, shifted)


def test_pan_profile_gives_memory_the_content_the_target_needs():
    memory = [pan_profile(f) for f in range(8, 12)]
    tail = [pan_profile(f) for f in range(20, 24)]
    target = [pan_profile(f) for f in range(24, 28)]
    assert min(memory) <= min(target) and max(target) <= max(memory)  # target view range lies inside memory's
    assert max(target) > max(tail)  # ...and exceeds what the previous chunk shows
    assert pan_profile(0) == pan_profile(5) == 0.0  # sink is elsewhere


def test_latent_index_is_chunk_aligned():
    assert latent_index(6.0) % 4 == 0 and latent_index(1.0) == 4
