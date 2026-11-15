"""WorldKV-style relevance retrieval from an unbounded archive of native KV.

Diagnostic, not a reproduction of WorldKV or DeCoPrune. Every clean (t=0) chunk
from latent frame 6 onward is archived on CPU as canonical pre-RoPE keys and
values. For each layer and chunk, archived frames are ranked by canonical
query/key relevance against the current chunk (64 pooled tokens per side, max over
key tokens, mean over query tokens and heads). The top `retained_frames` replace
the oldest frames of the attended window. No weights, sampler or schedule change.

rope="virtual": retrieved keys are re-indexed to consecutive positions just
before the recent tail (WorldTrace/WorldKV style). rope="native": each keeps its
original absolute temporal index.
"""
import torch
import torch.nn.functional as F

from worldsync.lingbot_retrieval import PoseKVMemory

POOL_TOKENS = 64


def pool_tokens(raw, heads, tokens=POOL_TOKENS):
    """[1, s, heads*d] -> L2-normalised [tokens, heads, d] (float32)."""
    x = raw[0].float()
    s = x.shape[0]
    x = F.adaptive_avg_pool1d(x.t().unsqueeze(0), tokens)[0].t()
    x = x.reshape(tokens, heads, -1)
    return F.normalize(x, dim=-1)


def rank_frames(query_pool, key_pools):
    """query_pool [Q,H,D], key_pools [F,K,H,D] -> relevance [F]."""
    sim = torch.einsum("qhd,fkhd->fhqk", query_pool, key_pools)
    return sim.max(dim=-1).values.mean(dim=(1, 2))


class RelevanceKVMemory(PoseKVMemory):
    def __init__(self, pipeline, retained_frames=4, rope="virtual", tail_frames=8, sink_frames=6):
        super().__init__(pipeline)
        if rope not in ("virtual", "native"):
            raise ValueError("rope must be virtual or native")
        self.retained = retained_frames
        self.rope = rope
        self.tail = tail_frames
        self.sink = sink_frames
        self.pooled = {}
        self.raw_query = None
        self.heads = None
        for block in pipeline.model.blocks:
            attention = block.self_attn
            self.heads = attention.num_heads
            self.handles.append(attention.norm_q.register_forward_hook(self._capture_query))

    def _capture_query(self, module, args, result):
        self.raw_query = result

    def begin(self, chunk, poses):
        self.chunk = chunk
        self.cached.clear()
        self.trace.append({"chunk": chunk, "retrieved_frame_ids": [], "archive_frame_ids": sorted(self.available),
                           "attention_replaced": False})

    def attend(self, query, key, value, **kwargs):
        if self.active is None:
            return self.original_attention(query, key, value, **kwargs)
        layer, start, frame_tokens, grid, freqs = self.active
        first_frame = start // frame_tokens
        bank = self.banks.setdefault(layer, {})
        pooled = self.pooled.setdefault(layer, {})
        if self.clean:
            if self.raw_key is None:
                raise RuntimeError("canonical key capture missing")
            raw = self.raw_key.view_as(query)
            v_current = value[:, -query.shape[1]:]
            for offset in range(query.shape[1] // frame_tokens):
                index = first_frame + offset
                if index < self.sink:
                    continue
                lo, hi = offset * frame_tokens, (offset + 1) * frame_tokens
                bank[index] = (raw[:, lo:hi].detach().to("cpu", copy=True),
                               v_current[:, lo:hi].detach().to("cpu", copy=True))
                pooled[index] = pool_tokens(raw[:, lo:hi].reshape(1, frame_tokens, -1), self.heads).to(torch.bfloat16)
                if layer == 0:
                    self.available.add(index)
        local_frames = key.shape[1] // frame_tokens
        # Only act once the stock window is full; budget is sink + retained + tail.
        if key.shape[1] % frame_tokens or local_frames != self.sink + 12:
            return self.original_attention(query, key, value, **kwargs)
        if layer not in self.cached:
            self.cached[layer] = self._retrieve(query, key, frame_tokens, grid, freqs, first_frame, layer)
        entry = self.cached[layer]
        if entry is None:
            return self.original_attention(query, key, value, **kwargs)
        retrieved_key, retrieved_value, chosen = entry
        combined_key = torch.cat([key[:, :self.sink * frame_tokens], retrieved_key, key[:, -self.tail * frame_tokens:]], dim=1)
        combined_value = torch.cat([value[:, :self.sink * frame_tokens], retrieved_value, value[:, -self.tail * frame_tokens:]], dim=1)
        self.trace[-1]["attention_replaced"] = True
        if layer in (0, 20, 39):
            self.trace[-1].setdefault("selected_by_layer", {})[str(layer)] = chosen
        self.trace[-1]["retrieved_frame_ids"] = chosen if layer == 0 else self.trace[-1]["retrieved_frame_ids"]
        self.retrieval_calls += 1
        return self.original_attention(query, combined_key, combined_value, **kwargs)

    def _retrieve(self, query, key, frame_tokens, grid, freqs, first_frame, layer):
        pooled = self.pooled[layer]
        previous = self.tail - query.shape[1] // frame_tokens  # kept frames before the current chunk
        candidates = sorted(index for index in pooled if index < first_frame - previous)
        if len(candidates) < self.retained or first_frame - previous - self.retained < self.sink:
            return None
        raw_query = self.raw_query.view_as(query).reshape(1, query.shape[1], -1)
        query_pool = pool_tokens(raw_query, self.heads)
        key_pools = torch.stack([pooled[index] for index in candidates]).float()
        score = rank_frames(query_pool, key_pools)
        order = torch.argsort(score, descending=True)[:self.retained].tolist()
        chosen = sorted(candidates[i] for i in order)
        bank = self.banks[layer]
        canonical = torch.cat([bank[i][0] for i in chosen], dim=1).to(key.device)
        retrieved_value = torch.cat([bank[i][1] for i in chosen], dim=1).to(key.device)
        with torch.amp.autocast("cuda", enabled=False):
            if self.rope == "virtual":
                virtual_grid = grid.clone()
                virtual_grid[:, 0] = len(chosen)
                retrieved_key = self.module.causal_rope_apply(canonical, virtual_grid, freqs,
                    start_frame=first_frame - previous - len(chosen))
            else:
                parts = []
                single = grid.clone()
                single[:, 0] = 1
                for position, index in enumerate(chosen):
                    parts.append(self.module.causal_rope_apply(
                        canonical[:, position * frame_tokens:(position + 1) * frame_tokens], single, freqs, start_frame=index))
                retrieved_key = torch.cat(parts, dim=1)
        return retrieved_key.to(key.dtype), retrieved_value, chosen

    def metadata(self):
        return {"method": "WorldKV-style canonical query/key relevance retrieval; diagnostic, not paper reproduction",
                "retrieved_latent_frames": self.retained, "rope": self.rope, "sink_latent_frames": self.sink,
                "tail_latent_frames": self.tail, "attended_latent_frames": self.sink + self.retained + self.tail,
                "stock_attended_latent_frames": 18,
                "relevance": f"{POOL_TOKENS} pooled tokens, cosine, max over keys, mean over queries and heads, per layer",
                "archive": "all clean chunks from latent frame 6, CPU, canonical pre-RoPE keys and values",
                "archive_bytes": sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                                     for bank in self.banks.values() for k, v in bank.values()),
                "retrieval_attention_calls": self.retrieval_calls, "trace": self.trace}
