"""Reader identity, memory dependence, trainability and checkpoint compatibility."""
import copy

import pytest
import torch
from torch import nn

from worldsync.lingbot_memory_reader import MemoryReader, attach, save, load


def test_zero_initialization_and_gradient_flow():
    torch.manual_seed(8)
    reader = MemoryReader(8, 4)
    q, k, v = torch.randn(2, 3, 2, 8), torch.randn(2, 5, 2, 8), torch.randn(2, 5, 2, 8)
    result = reader(q, k, v)
    assert torch.equal(result, torch.zeros_like(q))
    (result - 1).square().mean().backward()
    assert reader.out.weight.grad.abs().sum() > 0
    # A zero output projection initially blocks upstream gradients, as expected.
    with torch.no_grad():
        reader.out.weight.add_(-reader.out.weight.grad)
    reader.zero_grad()
    reader(q, k, v).square().mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in reader.parameters())


def test_reader_learns_memory_dependent_values_on_heldout_inputs():
    torch.manual_seed(1)
    reader = MemoryReader(8, 8)
    optimizer = torch.optim.Adam(reader.parameters(), lr=.03)
    # Single-slot associative recall: targets change with memory, never query.
    for _ in range(160):
        q, k, v = (torch.randn(8, 1, 1, 8) for _ in range(3))
        optimizer.zero_grad()
        (reader(q, k, v) - v).square().mean().backward()
        optimizer.step()
    q, k, v = (torch.randn(32, 1, 1, 8) for _ in range(3))
    correct = (reader(q, k, v) - v).square().mean()
    shuffled = (reader(q, k, v.flip(0)) - v).square().mean()
    assert correct < .03
    assert shuffled > 10 * correct


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        block = nn.Module()
        block.self_attn = nn.Module()
        block.self_attn.head_dim = 8
        block.self_attn.q = nn.Linear(8, 8)
        self.blocks = nn.ModuleList([block])


def test_only_reader_trains_and_roundtrip_matches(tmp_path):
    model = Tiny()
    baseline = copy.deepcopy(model)
    trainable = attach(model, 4)
    assert len(trainable) == 4
    assert all('memory_reader' in name for name, p in model.named_parameters() if p.requires_grad)
    with torch.no_grad():
        model.blocks[0].self_attn.memory_reader.out.weight.normal_()
    path = tmp_path / 'reader.pt'
    digest = save(model, path)
    assert load(baseline, path)['sha256'] == digest
    q, k, v = (torch.randn(1, 3, 2, 8) for _ in range(3))
    assert torch.equal(model.blocks[0].self_attn.memory_reader(q,k,v), baseline.blocks[0].self_attn.memory_reader(q,k,v))
    with pytest.raises(ValueError, match='already attached'):
        attach(model, 4)
    incompatible = Tiny()
    incompatible.blocks[0].self_attn.head_dim = 16
    with pytest.raises(ValueError, match='does not match'):
        load(incompatible, path)


def test_reader_is_connected_to_native_training_block(monkeypatch):
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    source = Path(__file__).resolve().parents[1] / 'external/LingBot-World2'
    if not source.exists():
        pytest.skip('pinned upstream source absent')
    existing = sys.modules.get('wan')
    if existing and not Path(existing.__file__).resolve().is_relative_to(source.resolve()):
        import subprocess
        subprocess.run([sys.executable, '-m', 'pytest', '-q',
                        f'{__file__}::test_reader_is_connected_to_native_training_block'], check=True)
        return
    monkeypatch.syspath_prepend(str(source))
    from wan.modules import model_fast
    from worldsync.matrix_attention import sdpa_attention
    from worldsync.lingbot_memory_train import layer_forward
    monkeypatch.setattr(model_fast, 'flash_attention', sdpa_attention)
    torch.manual_seed(4)
    model = nn.Module()
    block = model_fast.CausalWanAttentionBlock(64, 128, 4)
    model.blocks = nn.ModuleList([block])
    group = SimpleNamespace(e0=torch.zeros(1, 1, 6, 64), h=2, w=2,
                            positions=list(range(24,28)), cam=torch.randn(1,16,64))
    angles = torch.arange(1024)[:,None] * torch.linspace(.001,.1,8)[None]
    freqs = torch.polar(torch.ones_like(angles), angles).to(torch.complex128)
    x, text = torch.randn(1,16,64), torch.randn(1,4,64)
    kv = (torch.randn(1,56,4,16),torch.randn(1,56,4,16))
    baseline = layer_forward(block,x,group,freqs,text,kv)[0]
    parameters = attach(model,4)
    initial = layer_forward(block,x,group,freqs,text,kv)[0]
    torch.testing.assert_close(initial,baseline,atol=0,rtol=0)
    initial.square().mean().backward()
    reader = block.self_attn.memory_reader
    assert reader.out.weight.grad.abs().sum() > 0
    assert all(p.grad is None for name,p in model.named_parameters() if 'memory_reader' not in name)
    with torch.no_grad():
        reader.out.weight.add_(-.1*reader.out.weight.grad)
    trained = layer_forward(block,x,group,freqs,text,kv)[0]
    assert not torch.equal(initial,trained)
    # Removing the memory group gives sink6 + tail4, so no reader is used.
    no_memory = (kv[0][:,:40],kv[1][:,:40])
    with_reader = layer_forward(block,x,group,freqs,text,no_memory)[0]
    del block.self_attn.memory_reader
    without_reader = layer_forward(block,x,group,freqs,text,no_memory)[0]
    torch.testing.assert_close(with_reader,without_reader,atol=0,rtol=0)
