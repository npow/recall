"""Default-off fp8 compression of LingBot's rolling self-attention KV cache.

The causal cache stores bf16 K and V per layer: 40 layers x window x 1560 tokens x 5120 dims x 2 B x 2 (K, V)
is ~1.47 GB per latent frame, so a 46-frame window holds ~59 GB and a 70-frame window ~89 GB.
This stores K/V as float8_e4m3fn with one bf16 scale per (token, head) (+1.6% overhead) and dequantizes the
attended window to bf16 right before attention, one layer at a time. Weights stay bf16.

It replaces only the cache read/write/eviction lines of `CausalWanSelfAttention.forward`; the maths around
them (qk-norm, RoPE, attention, output projection) is the upstream code. Numerics are NOT bit-exact with bf16.
An optional audit keeps a bf16 shadow cache on a few layers and records the attention-output relative error.
Combining it with other memory hooks that read `kv_cache` is unsupported (they would see fp8).
"""
import math

import torch

FP8_MAX = 448.0  # largest finite float8_e4m3fn value
FP8 = torch.float8_e4m3fn


def quantize(x):
    """x [B, S, N, D] -> (fp8 values [B, S, N, D], bf16 scale [B, S, N, 1]); scale per token and head."""
    amax = x.detach().abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-8)
    scale = amax / FP8_MAX
    q = (x.float() / scale).clamp_(-FP8_MAX, FP8_MAX).to(FP8)
    return q, scale.to(torch.bfloat16)


def dequantize(q, scale, dtype=torch.bfloat16):
    return q.to(dtype) * scale.to(dtype)


def allocate(num_layers, shape, dtype, device, audit_layers=()):
    """Drop-in for pipeline._initialize_self_kv_cache with fp8 storage."""
    scale_shape = tuple(shape[:-1]) + (1,)
    caches = []
    for layer in range(num_layers):
        entry = {"k": torch.zeros(shape, dtype=FP8, device=device), "v": torch.zeros(shape, dtype=FP8, device=device),
                 "ks": torch.zeros(scale_shape, dtype=torch.bfloat16, device=device),
                 "vs": torch.zeros(scale_shape, dtype=torch.bfloat16, device=device),
                 "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                 "local_end_index": torch.tensor([0], dtype=torch.long, device=device)}
        if layer in audit_layers:
            entry["k_ref"] = torch.zeros(shape, dtype=dtype, device=device)
            entry["v_ref"] = torch.zeros(shape, dtype=dtype, device=device)
        caches.append(entry)
    return caches


def store(cache, start, end, key, value):
    cache["k"][:, start:end], cache["ks"][:, start:end] = quantize(key)
    cache["v"][:, start:end], cache["vs"][:, start:end] = quantize(value)
    if "k_ref" in cache:
        cache["k_ref"][:, start:end] = key
        cache["v_ref"][:, start:end] = value


def roll(cache, dst0, dst1, src0, src1):
    for name in ("k", "ks", "v", "vs") + (("k_ref", "v_ref") if "k_ref" in cache else ()):
        cache[name][:, dst0:dst1] = cache[name][:, src0:src1].clone()


def read(cache, lo, hi, dtype=torch.bfloat16):
    return (dequantize(cache["k"][:, lo:hi], cache["ks"][:, lo:hi], dtype),
            dequantize(cache["v"][:, lo:hi], cache["vs"][:, lo:hi], dtype))


class Fp8KVCache:
    def __init__(self, pipeline, audit_layers=(), quantize_current=False):
        from wan.modules import model_fast
        self.module = model_fast
        self.pipeline = pipeline
        self.audit_layers = set(audit_layers)
        self.quantize_current = quantize_current
        self.original_forward = model_fast.CausalWanSelfAttention.forward
        self.original_alloc = getattr(pipeline, "_initialize_self_kv_cache", None)
        self.layer_of = {id(block.self_attn): index for index, block in enumerate(pipeline.model.blocks)}
        self.errors = {}
        self.dequantized_bytes = 0
        owner = self

        def forward(self_attn, x, seq_lens, grid_sizes, freqs, kv_cache=None, current_start=0,
                    max_attention_size=1_000_000, frame_seqlen=None, seq_lens_int=None):
            if kv_cache is None or "ks" not in kv_cache:
                return owner.original_forward(self_attn, x, seq_lens, grid_sizes, freqs, kv_cache, current_start,
                                              max_attention_size, frame_seqlen, seq_lens_int)
            return owner.forward(self_attn, x, seq_lens, grid_sizes, freqs, kv_cache, current_start,
                                 max_attention_size, frame_seqlen)

        model_fast.CausalWanSelfAttention.forward = forward
        pipeline._initialize_self_kv_cache = lambda num_layers, shape, dtype, device: allocate(
            num_layers, shape, dtype, device, self.audit_layers)

    def forward(self, attn, x, seq_lens, grid_sizes, freqs, kv_cache, current_start, max_attention_size, frame_seqlen):
        mf = self.module
        b, s, n, d = *x.shape[:2], attn.num_heads, attn.head_dim
        q = attn.norm_q(attn.q(x)).view(b, s, n, d)
        k = attn.norm_k(attn.k(x)).view(b, s, n, d)
        v = attn.v(x).view(b, s, n, d)
        if frame_seqlen is None:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        start_frame = current_start // frame_seqlen
        roped_query = mf.causal_rope_apply(q, grid_sizes, freqs, start_frame=start_frame).type_as(v)
        roped_key = mf.causal_rope_apply(k, grid_sizes, freqs, start_frame=start_frame).type_as(v)
        current_end = current_start + roped_query.shape[1]
        sink_tokens = attn.sink_size * frame_seqlen
        cache_size = kv_cache["k"].shape[1]
        new_tokens = roped_query.shape[1]
        if attn.local_attn_size == -1:
            local_end, local_start = current_end, current_start
            store(kv_cache, local_start, local_end, roped_key, v)
        elif (current_end > kv_cache["global_end_index"].item()) and (
                new_tokens + kv_cache["local_end_index"].item() > cache_size):
            evicted = new_tokens + kv_cache["local_end_index"].item() - cache_size
            rolled = kv_cache["local_end_index"].item() - evicted - sink_tokens
            roll(kv_cache, sink_tokens, sink_tokens + rolled, sink_tokens + evicted, sink_tokens + evicted + rolled)
            local_end = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item() - evicted
            local_start = local_end - new_tokens
            store(kv_cache, local_start, local_end, roped_key, v)
        else:
            local_end = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
            local_start = local_end - new_tokens
            store(kv_cache, local_start, local_end, roped_key, v)
        lo = max(0, local_end - max_attention_size)
        if self.quantize_current:
            k_window, v_window = read(kv_cache, lo, local_end, roped_query.dtype)
        elif lo < local_start:
            # Past frames come from the fp8 cache; the chunk being generated attends to its own exact bf16 K/V.
            past_k, past_v = read(kv_cache, lo, local_start, roped_query.dtype)
            k_window = torch.cat([past_k, roped_key], dim=1)
            v_window = torch.cat([past_v, v], dim=1)
        else:
            k_window, v_window = roped_key[:, lo - local_start:], v[:, lo - local_start:]
        self.dequantized_bytes += (k_window.numel() + v_window.numel()) * k_window.element_size()
        out = mf.attention(roped_query, k_window, v_window)
        if "k_ref" in kv_cache:
            ref = mf.attention(roped_query, kv_cache["k_ref"][:, lo:local_end], kv_cache["v_ref"][:, lo:local_end])
            error = (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)
            self.errors.setdefault(self.layer_of.get(id(attn), -1), []).append(float(error))
        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end)
        return attn.o(out.flatten(2))

    clean = False

    def begin(self, chunk, poses):
        """Rollout hook; fp8 storage needs no per-chunk state."""

    def release_read_cache(self):
        """Rollout hook; nothing to release."""

    def metadata(self):
        audit = {str(layer): {"calls": len(values), "mean_rel_err": sum(values) / len(values), "max_rel_err": max(values)}
                 for layer, values in sorted(self.errors.items())}
        return {"method": "fp8 (e4m3) KV cache, per-token-per-head bf16 scales, dequantized to bf16 before attention",
                "weights": "bf16 (unquantized)", "current_chunk_attends_exact_bf16": not self.quantize_current, "bit_exact_with_bf16": False, "attention_audit_layers": sorted(self.audit_layers),
                "attention_output_relative_error": audit, "dequantized_gb": self.dequantized_bytes / 1e9}

    def close(self):
        self.module.CausalWanSelfAttention.forward = self.original_forward
        if self.original_alloc is not None:
            # Restore the bound method: deleting the instance attribute re-exposes the class method.
            try:
                del self.pipeline._initialize_self_kv_cache
            except AttributeError:
                pass
