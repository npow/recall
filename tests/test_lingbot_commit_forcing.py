import copy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
from worldsync.lingbot_commit_forcing import CommitForcing


@pytest.fixture(autouse=True)
def one_cpu_thread():
    before=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def native_module(monkeypatch,request):
    source=Path(__file__).resolve().parents[1]/'external/LingBot-World2'
    if not source.exists():pytest.skip('pinned native source unavailable')
    existing=sys.modules.get('wan')
    if existing and (getattr(existing,'__file__',None) is None or not Path(existing.__file__).resolve().is_relative_to(source.resolve())):
        import subprocess
        subprocess.run([sys.executable,'-m','pytest','-q',request.node.nodeid],check=True)
        return None
    monkeypatch.syspath_prepend(str(source))
    from wan.modules import model_fast
    from worldsync.matrix_attention import sdpa_attention
    monkeypatch.setattr(model_fast,'attention',sdpa_attention)
    return model_fast


def fixture_model(native,dtype):
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn=native.CausalWanSelfAttention(64,4,local_attn_size=18,sink_size=6)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks=torch.nn.ModuleList([Block(),Block()])
        def forward(self,x,t,caches,start,freqs):
            for block,cache in zip(self.blocks,caches):
                x=x+block.self_attn(x,torch.tensor([16]),torch.tensor([[4,2,2]]),freqs,
                                   cache,start,72,frame_seqlen=4)
            return x
    torch.manual_seed(194)
    model=Model().eval().to(dtype)
    frequency=torch.arange(1024)[:,None]*torch.linspace(.001,.1,8)[None]
    freqs=torch.polar(torch.ones_like(frequency),frequency).to(torch.complex128)
    def caches():
        return [{'k':torch.zeros(1,72,4,16,dtype=dtype),'v':torch.zeros(1,72,4,16,dtype=dtype),
                 'global_end_index':torch.tensor([0]),'local_end_index':torch.tensor([0])}for _ in range(2)]
    return model,freqs,caches


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_actual_native_clean_reads_current_and_predecessor_only(monkeypatch,dtype,request):
    native=native_module(monkeypatch,request)
    if native is None:return
    model,freqs,make_caches=fixture_model(native,dtype)
    baseline=copy.deepcopy(model)
    inputs=[[torch.randn(1,16,64).to(dtype) for _ in range(5)]for _ in range(8)]
    results=[]
    original_global=native.attention
    original_forward=native.CausalWanSelfAttention.forward
    for enabled,candidate in [(False,baseline),(True,model)]:
        caches=make_caches()
        values=[]
        helper=CommitForcing(SimpleNamespace(model=candidate),enabled=enabled)
        with helper,torch.no_grad():
            for chunk,steps in enumerate(inputs):
                for step,x in enumerate(steps):
                    clean=step==4
                    before=[(c['k'].clone(),c['v'].clone())for c in caches]
                    output=candidate(x,t=torch.tensor([0. if clean else 500.]),caches=caches,
                                     start=chunk*16,freqs=freqs)
                    assert not helper.clean
                    if clean:
                        for cache,(key,value)in zip(caches,before):
                            end=int(cache['local_end_index'])
                            assert torch.equal(cache['k'][:,:end-16],key[:,:end-16])
                            assert torch.equal(cache['v'][:,:end-16],value[:,:end-16])
                    values.append((output.clone(),[c['k'].clone()for c in caches]))
        assert all('forward' not in b.self_attn.__dict__ for b in candidate.blocks)
        assert not candidate._forward_pre_hooks and not candidate._forward_hooks
        assert native.attention is original_global
        assert native.CausalWanSelfAttention.forward is original_forward
        results.append((values,helper.trace))
    # Two initial blocks have no older history to exclude. The third block's
    # denoising is still unchanged; its clean write first changes deeper KV.
    for i in range(14):
        assert torch.equal(results[0][0][i][0],results[1][0][i][0])
    clean_third=14
    assert not torch.equal(results[0][0][clean_third][0],results[1][0][clean_third][0])
    assert torch.equal(results[0][0][clean_third][1][0],results[1][0][clean_third][1][0])
    assert not torch.equal(results[0][0][clean_third][1][1],results[1][0][clean_third][1][1])
    trace=results[1][1]
    assert len(trace)==8*5*2
    assert [r['read_key_tokens']for r in trace if r['clean']and r['layer']==0]==[16,32,32,32,32,32,32,32]
    assert all(r['read_key_tokens']==r['available_key_tokens']for r in trace if not r['clean'])
    assert max(r['available_key_tokens']for r in trace)==72


def test_instance_isolation_and_exception_cleanup(monkeypatch,request):
    native=native_module(monkeypatch,request)
    if native is None:return
    model,freqs,make_caches=fixture_model(native,torch.float32)
    other=copy.deepcopy(model)
    method=native.CausalWanSelfAttention.forward
    attention=native.attention
    with pytest.raises(RuntimeError,match='intentional'):
        with CommitForcing(SimpleNamespace(model=model),enabled=True)as helper:
            assert all('forward' in b.self_attn.__dict__ for b in model.blocks)
            assert all('forward' not in b.self_attn.__dict__ for b in other.blocks)
            with torch.no_grad():model(torch.zeros(1,16,64),t=torch.tensor([0.]),caches=make_caches(),start=0,freqs=freqs)
            raise RuntimeError('intentional')
    assert helper.closed and not helper.clean
    assert all('forward' not in b.self_attn.__dict__ for b in model.blocks)
    assert not model._forward_pre_hooks and not model._forward_hooks
    assert native.CausalWanSelfAttention.forward is method and native.attention is attention


def test_partial_attachment_failure_preserves_other_wrapper(monkeypatch,request):
    native=native_module(monkeypatch,request)
    if native is None:return
    model,_,_=fixture_model(native,torch.float32)
    second=model.blocks[1].self_attn
    previous=second.forward
    second.forward=previous
    helper=CommitForcing(SimpleNamespace(model=model),enabled=True)
    with pytest.raises(ValueError,match='unwrapped'):
        helper.__enter__()
    assert 'forward' not in model.blocks[0].self_attn.__dict__
    assert second.__dict__['forward'] is previous
    assert not model._forward_pre_hooks and not model._forward_hooks
    assert helper.closed


def test_mixed_noise_levels_keep_full_context(monkeypatch,request):
    native=native_module(monkeypatch,request)
    if native is None:return
    model,freqs,make_caches=fixture_model(native,torch.float32)
    caches=make_caches()
    versions={name:p._version for name,p in model.named_parameters()}
    with CommitForcing(SimpleNamespace(model=model),enabled=True)as helper,torch.no_grad():
        assert all(b.self_attn.forward.__func__.__code__ is native.CausalWanSelfAttention.forward.__code__ for b in model.blocks)
        for chunk in range(4):
            model(torch.randn(1,16,64),t=torch.tensor([500.]),caches=caches,start=chunk*16,freqs=freqs)
        model(torch.randn(1,16,64),t=torch.tensor([0.,500.]),caches=caches,start=4*16,freqs=freqs)
        assert all(not row['clean'] and row['read_key_tokens']==72 for row in helper.trace[-2:])
    assert versions=={name:p._version for name,p in model.named_parameters()}


@pytest.mark.parametrize('late_change',['attention','reader'])
def test_later_memory_attachment_fails_before_forward(monkeypatch,request,late_change):
    native=native_module(monkeypatch,request)
    if native is None:return
    model,freqs,make_caches=fixture_model(native,torch.float32)
    with CommitForcing(SimpleNamespace(model=model),enabled=True)as helper:
        with monkeypatch.context()as patch:
            if late_change=='attention':patch.setattr(native,'attention',lambda *a,**k:None)
            else:patch.setattr(model.blocks[0].self_attn,'memory_reader',torch.nn.Identity(),raising=False)
            with pytest.raises(ValueError,match='isolated'):
                model(torch.zeros(1,16,64),t=torch.tensor([0.]),caches=make_caches(),start=0,freqs=freqs)
            assert not helper.clean
    assert all('forward' not in b.self_attn.__dict__ for b in model.blocks)
