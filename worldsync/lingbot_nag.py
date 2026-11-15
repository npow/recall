"""Normalized Attention Guidance (NAG, arXiv 2505.21179) for the CFG-free LingBot causal-fast model.

EXPERIMENTAL, default off, untested upstream on a causal KV-cached model. The distilled model runs
without classifier-free guidance, so a negative prompt has no effect. NAG restores one by evaluating
every cross-attention layer a second time with the negative-prompt context and extrapolating in
attention-output space:

    z_ext = z_pos + scale * (z_pos - z_neg)
    ratio = ||z_ext||_1 / ||z_pos||_1            (per token)
    z_hat = z_ext * min(1, tau / ratio)          (norm clipping)
    z_out = alpha * z_hat + (1 - alpha) * z_pos  (blend)

`install` wraps `WanCrossAttention.forward` of every block; `uninstall` restores it. The negative
keys/values are computed once per installation from the model's own text embedding of the negative
prompt. The positive path (including the cross-attention K/V cache) is untouched.
"""
import torch

DEFAULT_NEGATIVE = "static, still, frozen, motionless, paused, nothing moves, 静态, 静止不动的画面"


def nag_combine(z_pos, z_neg, scale, tau, alpha):
    z_ext = z_pos + scale * (z_pos - z_neg)
    norm_pos = z_pos.float().abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
    norm_ext = z_ext.float().abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
    ratio = norm_ext / norm_pos
    z_hat = z_ext * torch.clamp(tau / ratio, max=1.0).to(z_ext.dtype)
    return alpha * z_hat + (1 - alpha) * z_pos


class NAG:
    def __init__(self, pipeline, negative=DEFAULT_NEGATIVE, scale=4.0, tau=2.5, alpha=0.25):
        self.pipeline, self.negative = pipeline, negative
        self.scale, self.tau, self.alpha = float(scale), float(tau), float(alpha)
        self.model = pipeline.model
        self.raw = None           # T5 output for the negative prompt, [L, C_text]
        self.embedded = None      # model.text_embedding(padded raw), [1, text_len, dim]
        self.kv = {}
        self.originals = []
        self.calls = 0

    def install(self):
        self.pipeline.text_encoder.model.to(self.pipeline.device)
        self.raw = self.pipeline.text_encoder([self.negative], self.pipeline.device)[0]
        import wan.modules.model_fast as model_fast
        for index, block in enumerate(self.model.blocks):
            attention = block.cross_attn
            original = attention.forward
            self.originals.append((attention, original))

            def forward(x, context, context_lens, crossattn_cache=None, cross_attn_first_call=None,
                        _attention=attention, _original=original, _index=index):
                z_pos = _original(x, context, context_lens, crossattn_cache=crossattn_cache,
                                  cross_attn_first_call=cross_attn_first_call)
                b, n, d = x.size(0), _attention.num_heads, _attention.head_dim
                if _index not in self.kv:
                    if self.embedded is None:
                        raw = self.raw.to(context.dtype)
                        padded = torch.cat([raw, raw.new_zeros(self.model.text_len - raw.size(0), raw.size(1))])
                        self.embedded = self.model.text_embedding(padded[None])
                    negative = self.embedded.to(context.dtype)
                    self.kv[_index] = (_attention.norm_k(_attention.k(negative)).view(1, -1, n, d),
                                       _attention.v(negative).view(1, -1, n, d))
                k, v = self.kv[_index]
                q = _attention.norm_q(_attention.q(x)).view(b, -1, n, d)
                z_neg = _attention.o(model_fast.flash_attention(q, k.expand(b, -1, -1, -1), v.expand(b, -1, -1, -1),
                                                              k_lens=context_lens).flatten(2))
                self.calls += 1
                return nag_combine(z_pos, z_neg, self.scale, self.tau, self.alpha)

            attention.forward = forward
        return self

    def uninstall(self):
        for attention, original in self.originals:
            attention.forward = original
        self.originals, self.kv, self.embedded = [], {}, None

    def metadata(self):
        return {"method": "Normalized Attention Guidance in cross-attention (experimental)", "negative_prompt": self.negative,
                "scale": self.scale, "tau": self.tau, "alpha": self.alpha, "cross_attention_calls": self.calls}
