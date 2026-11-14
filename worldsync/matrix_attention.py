"""SDPA compatibility for stock Matrix's direct varlen FlashAttention calls.

The upstream --fa_version 0 dispatcher is not used by the model. This bridge
preserves the supplied valid lengths; the unused upstream SDPA helper drops them.
"""
import torch
import torch.nn.functional as F


def sdpa_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                   softmax_scale=None, q_scale=None, causal=False,
                   window_size=(-1, -1), deterministic=False,
                   dtype=torch.bfloat16, version=None):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("expected [batch, sequence, heads, channels]")
    if window_size != (-1, -1) or causal:
        raise ValueError("stock Matrix uses noncausal global attention; unsupported mask requested")
    batch, queries = q.shape[:2]
    original_dtype = q.dtype
    compute_dtype = q.dtype if q.dtype in {torch.float16, torch.bfloat16} else dtype
    output = torch.zeros_like(q, dtype=original_dtype)
    for index in range(batch):
        nq = int(q_lens[index]) if q_lens is not None else queries
        nk = int(k_lens[index]) if k_lens is not None else k.shape[1]
        if not 0 < nq <= queries or not 0 < nk <= k.shape[1]:
            raise ValueError("invalid attention sequence lengths")
        qi = q[index:index + 1, :nq].transpose(1, 2).to(compute_dtype)
        ki = k[index:index + 1, :nk].transpose(1, 2).to(compute_dtype)
        vi = v[index:index + 1, :nk].transpose(1, 2).to(compute_dtype)
        if q_scale is not None:
            qi = qi * q_scale
        value = F.scaled_dot_product_attention(qi, ki, vi, dropout_p=dropout_p, scale=softmax_scale)
        output[index:index + 1, :nq] = value.transpose(1, 2).to(original_dtype)
    return output
