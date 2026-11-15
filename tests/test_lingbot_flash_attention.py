import pytest
import torch

from worldsync import lingbot_flash_attention as fa4
from worldsync.matrix_attention import sdpa_attention


def reference_kernel(q, k, v, softmax_scale, causal):
    assert not causal
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        scale=softmax_scale).transpose(1, 2), None


@pytest.mark.parametrize("batch,queries,keys", [(1, 5, 7), (2, 5, 7)])
def test_fa4_bridge_preserves_ragged_lengths_padding_and_scaling(monkeypatch, batch, queries, keys):
    monkeypatch.setattr(fa4, "kernel", lambda: reference_kernel)
    torch.manual_seed(9)
    q = torch.randn(batch, queries, 2, 16)
    k = torch.randn(batch, keys, 2, 16)
    v = torch.randn_like(k)
    q_lens = torch.tensor([queries] + [queries - 2] * (batch - 1))
    k_lens = torch.tensor([keys - 2] * batch)
    # Excluded keys must not leak into the attention result.
    k[:, -2:] = 1e4
    v[:, -2:] = 1e4
    kwargs = dict(q_lens=q_lens, k_lens=k_lens, softmax_scale=.1, q_scale=.5)
    actual = fa4.flash_attention(q, k, v, **kwargs)
    expected = sdpa_attention(q, k, v, **kwargs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.dtype == q.dtype
    if batch > 1:
        assert torch.count_nonzero(actual[1, queries - 2:]) == 0


@pytest.mark.parametrize("kwargs", [dict(causal=True), dict(window_size=(2, 2)),
                                    dict(dropout_p=.1), dict(k_lens=[0])])
def test_fa4_bridge_rejects_unsupported_or_empty_attention(kwargs):
    q = torch.ones(1, 4, 2, 16)
    with pytest.raises(ValueError):
        fa4.flash_attention(q, q, q, **kwargs)
