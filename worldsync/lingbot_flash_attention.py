"""Opt-in FA4 bridge for native Wan attention, preserving valid sequence lengths."""
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def kernel():
    # Loading stays lazy so established SDPA environments need no FA4 packages.
    from flash_attn.cute import flash_attn_func
    return flash_attn_func


def flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                    softmax_scale=None, q_scale=None, causal=False,
                    window_size=(-1, -1), deterministic=False,
                    dtype=torch.bfloat16, version=None):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("expected [batch, sequence, heads, channels]")
    if causal or window_size != (-1, -1) or dropout_p:
        raise ValueError("LingBot runtime bridge requires global noncausal attention without dropout")
    original_dtype = q.dtype
    compute_dtype = q.dtype if q.dtype in {torch.float16, torch.bfloat16} else dtype
    batch, queries = q.shape[:2]
    output = None
    for index in range(batch):
        nq = int(q_lens[index]) if q_lens is not None else queries
        nk = int(k_lens[index]) if k_lens is not None else k.shape[1]
        if not 0 < nq <= queries or not 0 < nk <= k.shape[1]:
            raise ValueError("invalid attention sequence lengths")
        qi = q[index:index + 1, :nq].to(compute_dtype)
        ki = k[index:index + 1, :nk].to(compute_dtype)
        vi = v[index:index + 1, :nk].to(compute_dtype)
        if q_scale is not None:
            qi = qi * q_scale
        result = kernel()(qi, ki, vi, softmax_scale=softmax_scale, causal=False)
        if isinstance(result, tuple):
            result = result[0]
        result = result.to(original_dtype)
        if batch == 1 and nq == queries:
            return result
        if output is None:
            output = torch.zeros_like(q, dtype=original_dtype)
        output[index:index + 1, :nq] = result
    return output
