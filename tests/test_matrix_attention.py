import pytest
import torch

from worldsync.matrix_attention import sdpa_attention


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_varlen_bridge_excludes_poisoned_padding_and_matches_attention_formula(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(7)
    q = torch.randn(2, 7, 3, 16, device=device, dtype=torch.bfloat16)
    k = torch.randn(2, 9, 3, 16, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q_lens, k_lens = [7, 4], [5, 3]
    for i, length in enumerate(k_lens):
        k[i, length:] = 100
        v[i, length:] = 100
    actual = sdpa_attention(q, k, v, q_lens=q_lens, k_lens=k_lens, softmax_scale=0.2)
    for i, (nq, nk) in enumerate(zip(q_lens, k_lens)):
        qi, ki, vi = [x.double().transpose(0, 1) for x in (q[i, :nq], k[i, :nk], v[i, :nk])]
        expected = ((qi @ ki.transpose(-1, -2) * 0.2).softmax(-1) @ vi).transpose(0, 1)
        torch.testing.assert_close(actual[i, :nq].double(), expected, atol=0.015, rtol=0.015)
        assert torch.count_nonzero(actual[i, nq:]) == 0
    altered = v.clone()
    for i, length in enumerate(k_lens):
        altered[i, length:] = -100
    assert torch.equal(actual, sdpa_attention(q, k, altered, q_lens=q_lens, k_lens=k_lens, softmax_scale=0.2))


def test_bridge_rejects_unimplemented_attention_masks():
    data = torch.zeros(1, 2, 1, 16)
    with pytest.raises(ValueError, match="unsupported mask"):
        sdpa_attention(data, data, data, causal=True)
