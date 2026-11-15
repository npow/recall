"""CPU math proof for native global cache versus an unevicted seventy-frame cache."""
import hashlib
from pathlib import Path
import sys

import pytest
import torch


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_global_cache_matches_seventy_frame_cache_before_eviction(monkeypatch,dtype):
    source=Path(__file__).resolve().parents[1]/'external/LingBot-World2'
    if not source.exists():pytest.skip('pinned native checkout not installed')
    existing=sys.modules.get('wan')
    if existing and not Path(existing.__file__).resolve().is_relative_to(source.resolve()):
        import subprocess
        label='dtype0' if dtype==torch.float32 else 'dtype1'
        subprocess.run([sys.executable,'-m','pytest','-q',f'{__file__}::{sys._getframe().f_code.co_name}[{label}]'],check=True)
        return
    monkeypatch.syspath_prepend(str(source))
    from wan.modules import model_fast
    from worldsync.matrix_attention import sdpa_attention
    monkeypatch.setattr(model_fast,'attention',sdpa_attention)
    import copy
    torch.manual_seed(813)
    layer=model_fast.CausalWanSelfAttention(64,4,local_attn_size=70,sink_size=6).eval().to(dtype)
    inputs=[[torch.randn(1,16,64).to(dtype) for _ in range(5)] for _ in range(13)]
    frequency=torch.arange(1024)[:,None]*torch.linspace(.001,.1,8)[None]
    freqs=torch.polar(torch.ones_like(frequency),frequency).to(torch.complex128)
    digests=[]
    for context in (70,-1):
        module=copy.deepcopy(layer)
        module.local_attn_size=context
        frames=70 if context==70 else 52
        cache={'k':torch.zeros(1,frames*4,4,16,dtype=dtype),'v':torch.zeros(1,frames*4,4,16,dtype=dtype),
               'global_end_index':torch.tensor([0]),'local_end_index':torch.tensor([0])}
        hashes=[]
        with torch.no_grad():
            for chunk,steps in enumerate(inputs):
                for x in steps:
                    output=module(x,torch.tensor([16]),torch.tensor([[4,2,2]]),freqs,cache,chunk*16,
                                  280 if context==70 else 1_000_000,frame_seqlen=4)
                    end=(chunk+1)*16
                    assert cache['global_end_index'].item()==end
                    assert cache['local_end_index'].item()==end
                    data=b''.join(t.contiguous().view(torch.uint8).numpy().tobytes()
                                  for t in (output,cache['k'][:,:end],cache['v'][:,:end]))
                    hashes.append(hashlib.sha256(data).hexdigest())
        digests.append(hashes)
    assert len(digests[0])==65
    assert digests[0]==digests[1]
