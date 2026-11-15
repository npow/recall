from pathlib import Path
import sys
import pytest
import torch
from worldsync.lingbot_vae_fixedpoint import encode_zero_tail,same_cache,snapshot_cache


def test_snapshot_detects_inplace_change_and_every_cache_tensor():
    cache=[torch.tensor([0.]),torch.tensor([4.]),None]
    saved=snapshot_cache(cache)
    cache[1].add_(1)
    assert not same_cache(saved,cache)
    cache[1].sub_(1)
    assert same_cache(saved,cache)
    negative_zero=[torch.tensor([-0.]),torch.tensor([4.]),None]
    assert not same_cache(saved,negative_zero)


@pytest.mark.parametrize('frames',[13,205,637])
def test_real_native_tiny_vae_preserves_all_normalized_bits(monkeypatch,frames):
    source=Path(__file__).resolve().parents[1]/'external/LingBot-World2'
    if not source.exists():pytest.skip('native source checkout unavailable')
    existing=sys.modules.get('wan')
    if existing and not Path(existing.__file__).resolve().is_relative_to(source.resolve()):
        import subprocess
        subprocess.run([sys.executable,'-m','pytest','-q',f'{__file__}::{sys._getframe().f_code.co_name}[{frames}]'],check=True)
        return
    monkeypatch.syspath_prepend(str(source))
    from wan.modules.vae2_1 import WanVAE_
    threads=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(183)
        model=WanVAE_(dim=4,z_dim=2,dim_mult=[1,2,4,4],num_res_blocks=1,
                      temperal_downsample=[False,True,True]).eval().requires_grad_(False)
        x=torch.zeros(1,3,frames,16,16)
        x[:,:,:1]=torch.randn(1,3,1,16,16)
        scale=[torch.tensor([.1,-.2]),torch.tensor([.7,1.3])]
        with torch.inference_mode():
            expected=model.encode(x,scale)
            info={}
            actual=encode_zero_tail(model,x,scale,stats=info)
        assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
        assert info['full_conv1_input_shape'][2]==1+(frames-1)//4
        assert all(v is None for v in model._enc_feat_map)
        if frames>=205:
            assert info['full_cache_bitwise_equal']
            assert info['repeated_groups']>0
            assert info['encoder_calls']<info['native_groups']
        x[:,:,-1]=.1
        with torch.inference_mode():
            info={}
            expected=model.encode(x,scale)
            actual=encode_zero_tail(model,x,scale,stats=info)
        assert info['status']=='native_fallback'
        assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
    finally:torch.set_num_threads(threads)


def test_constant_latent_with_changing_hidden_cache_is_not_a_fixed_point():
    class Encoder(torch.nn.Module):
        def forward(self,x,feat_cache,feat_idx):
            feat_cache[0].add_(1)
            feat_idx[0]+=1
            return torch.zeros(x.shape[0],2,1,x.shape[-2],x.shape[-1])
    class Model(torch.nn.Module):
        z_dim=1
        def __init__(self):
            super().__init__()
            self.encoder=Encoder()
            self.conv1=torch.nn.Identity()
        def clear_cache(self):
            self._enc_feat_map=[torch.zeros(1)]
            self._enc_conv_idx=[0]
        def encode(self,x,scale):
            raise AssertionError('Eligible zero-tail input should execute the checked loop')
    model=Model().eval()
    stats={}
    with torch.inference_mode():
        result=encode_zero_tail(model,torch.zeros(1,3,205,2,2),[0,1],stats=stats)
    assert torch.count_nonzero(result)==0
    assert stats['encoder_calls']==52
    assert stats['repeated_groups']==0
    assert not stats['full_cache_bitwise_equal']
