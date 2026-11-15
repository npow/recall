"""DeCoPrune (arXiv 2609.39096, github.com/DeCoPrune/CMBench) on LingBot-World2 14B causal-fast.

Source-faithful port of the released `decoprune` policy. The exact spec and every deviation are in
`reproduction/decoprune-spec.md`. Diagnostic only: not a PlayWorld score, not a paper reproduction.

Per chunk n >= 1 the live x0 prediction at denoising step `generation_step_index` is compared with the
chunk's final clean latent; each 2x2-patch KV token with MSE <= threshold is dropped from history once
its chunk is older than the two most recent chunks. Kept tokens persist (unbounded cache). Keys are
archived canonical (post-`norm_k`, pre-RoPE) and re-roped on read at true or re-indexed virtual
temporal positions (`all_bands`: last 8 frames true, older frames squeezed into a 9-frame span).
"""
import math

import torch

RECENT_FRAMES = 8
VIRTUAL_SPAN = 17.0
CHUNK_FRAMES = 4


def token_mse_scores(predicted, target):
    """[C,F,H,W] x2 -> per-token MSE, flattened in (frame, h, w) order, one token per 2x2 patch."""
    if tuple(predicted.shape) != tuple(target.shape):
        raise ValueError(f"x0/target shape mismatch: {tuple(predicted.shape)} vs {tuple(target.shape)}")
    values = (predicted.float() - target.float()).pow(2)
    channels, frames, height, width = values.shape
    if height % 2 or width % 2:
        raise ValueError(f"scores require even latent dimensions, got {height}x{width}")
    return values.reshape(channels, frames, height // 2, 2, width // 2, 2).mean(dim=(0, 3, 5)).reshape(-1)


def keep_mask(scores, threshold):
    """Keep iff score > threshold; non-finite scores are kept (an unstable score is not evidence)."""
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    return ~torch.isfinite(scores) | (scores > threshold)


def virtual_positions(q0, mode="all_bands", span=VIRTUAL_SPAN, recent=RECENT_FRAMES):
    """Temporal RoPE position for each of the q0 real frames before the current chunk.

    Released formula, except positions stay real for q0 <= span: the released code expands (and can go
    negative, q0=12 -> -5) when 8 < q0 < 17. See spec deviation 3.
    """
    real = torch.arange(q0, dtype=torch.float64)
    if mode == "off" or q0 <= span:
        return real
    offset = q0 - real
    scale = (span - recent) / (q0 - recent)
    virtual_offset = torch.where(offset <= recent, offset, recent + (offset - recent) * scale)
    return q0 - virtual_offset


def rope_tokens(x, frame_position, h_index, w_index, freqs):
    """RoPE for gathered tokens with fractional temporal positions.

    x [1,T,n,d] canonical keys; frame_position float64 [T]; h_index/w_index long [T]; freqs the model's
    complex table [1024, d/2]. Integer positions reproduce `causal_rope_apply` exactly because
    freqs[0][p] = exp(i*p*omega) and omega = angle(freqs[0][1]) (|omega| <= 1 < pi).
    """
    tokens, heads, half = x.shape[1], x.shape[2], x.shape[3] // 2
    parts = freqs.split([half - 2 * (half // 3), half // 3, half // 3], dim=1)
    omega = torch.angle(parts[0][1])
    phase = frame_position.to(omega.device)[:, None] * omega[None, :]
    temporal = torch.polar(torch.ones_like(phase), phase)
    multiplier = torch.cat([temporal, parts[1][h_index], parts[2][w_index]], dim=1).reshape(tokens, 1, -1)
    complex_x = torch.view_as_complex(x[0].to(torch.float64).reshape(tokens, heads, -1, 2))
    return torch.view_as_real(complex_x * multiplier).flatten(2)[None].type_as(x)


class DeCoPruneMemory:
    def __init__(self, pipeline, threshold=0.10, generation_step_index=2, rope="all_bands"):
        from wan.modules import model_fast
        if rope not in ("all_bands", "off"):
            raise ValueError("rope must be all_bands or off")
        self.pipeline = pipeline
        self.module = model_fast
        self.original_attention = model_fast.attention
        self.threshold = float(threshold)
        self.step_index = int(generation_step_index)
        self.rope = rope
        self.archive = {}        # layer -> chunk -> {"k","v","idx"}; idx None = whole chunk
        self.masks = {}          # chunk -> bool [tokens]
        self.records = {}        # chunk -> score statistics
        self.attended = {}       # (chunk, clean) -> history tokens attended in layer 0
        self.handles = []
        self.active = None
        self.raw_key = None
        self.clean = False
        self.chunk = -1
        self.calls = 0
        self.probe = None
        self.trace = []
        self.positions_cache = {}
        self.geometry = None
        self.tokens = None
        self.token_grid = None
        self.attention_calls = 0
        for layer, block in enumerate(pipeline.model.blocks):
            attention = block.self_attn

            def enter(module, args, kwargs, layer=layer):
                self.active = (layer, int(args[5]), int(kwargs["frame_seqlen"]), args[2], args[3])
                self.raw_key = None

            def capture(module, args, result):
                self.raw_key = result

            def leave(module, args, result):
                self.active = None
                self.raw_key = None
            self.handles.extend([attention.register_forward_pre_hook(enter, with_kwargs=True),
                                 attention.norm_k.register_forward_hook(capture),
                                 attention.register_forward_hook(leave)])
        self.handles.append(pipeline.model.register_forward_hook(self._after_model, with_kwargs=True))
        self.original_convert = pipeline._convert_flow_pred_to_x0

        def convert(*args, **kwargs):
            x0 = self.original_convert(*args, **kwargs)
            if self.calls == self.step_index:
                self.probe = x0.detach().clone()
            self.calls += 1
            return x0
        pipeline._convert_flow_pred_to_x0 = convert
        model_fast.attention = self.attend

    # -- rollout interface (shared with PoseKVMemory) -------------------------------------------
    def begin(self, chunk, poses):
        self.chunk = chunk
        self.calls = 0
        self.probe = None
        self.trace.append({"chunk": chunk})

    def _after_model(self, module, args, kwargs, result):
        if float(kwargs["t"].flatten()[0]) != 0:
            return
        chunk = int(kwargs["current_start"]) // (CHUNK_FRAMES * int(kwargs["frame_seqlen"]))
        if chunk < 1:
            return  # chunk 0 is never scored (context_probe returns None) and stays whole
        if self.probe is None:
            raise RuntimeError("probe x0 was not captured before the clean pass")
        scores = token_mse_scores(self.probe, kwargs["x"][0])
        mask = keep_mask(scores, self.threshold)
        self.masks[chunk] = mask
        finite = scores[torch.isfinite(scores)]
        self.records[chunk] = {"chunk": chunk, "tokens": int(mask.numel()), "kept": int(mask.sum()),
            "kept_fraction": float(mask.float().mean()), "score_mean": float(finite.mean()),
            "score_median": float(finite.median()), "score_p90": float(finite.quantile(0.9)),
            "score_max": float(finite.max())}
        self.probe = None

    def _positions(self, n, device):
        key = (n, self.rope)
        if key not in self.positions_cache:
            self.positions_cache[key] = virtual_positions(CHUNK_FRAMES * n, self.rope).to(device)
        return self.positions_cache[key]

    def _history(self, layer, n, cutoff, freqs, frame_tokens, dtype, device):
        archive = self.archive[layer]
        positions = self._positions(n, device)
        grid_h, grid_w = self.token_grid
        keys, values, frames, ids = [], [], [], []
        full_index = torch.arange(archive[0]["k"].shape[1], device=device)
        for chunk in range(n):
            entry = archive[chunk]
            if 1 <= chunk <= cutoff and entry["idx"] is None:
                mask = self.masks[chunk]
                entry["k"], entry["v"] = entry["k"][:, mask], entry["v"][:, mask]
                entry["idx"] = mask.nonzero().squeeze(1)
            index = full_index if entry["idx"] is None else entry["idx"]
            keys.append(entry["k"])
            values.append(entry["v"])
            ids.append(index)
            frames.append(positions[CHUNK_FRAMES * chunk + index // frame_tokens])
        index = torch.cat(ids)
        within = index % frame_tokens
        key = rope_tokens(torch.cat(keys, dim=1), torch.cat(frames), within // grid_w, within % grid_w, freqs).to(dtype)
        return key, torch.cat(values, dim=1), int(index.numel())

    def attend(self, query, key, value, **kwargs):
        if self.active is None:
            return self.original_attention(query, key, value, **kwargs)
        layer, start, frame_tokens, grid, freqs = self.active
        tokens = query.shape[1]
        chunk = start // tokens
        if self.tokens is None:
            self.tokens = tokens
            self.token_grid = (int(grid[0, 1]), int(grid[0, 2]))
        if self.clean:
            if self.raw_key is None:
                raise RuntimeError("canonical key capture missing")
            self.archive.setdefault(layer, {})[chunk] = {
                "k": self.raw_key.view_as(query).detach().clone(),
                "v": value[:, -tokens:].detach().clone(), "idx": None}
        if chunk == 0:
            return self.original_attention(query, key, value, **kwargs)
        cutoff = chunk - 2 if self.clean else chunk - 3
        history_key, history_value, count = self._history(
            layer, chunk, cutoff, freqs, frame_tokens, key.dtype, key.device)
        if layer == 0:
            self.attended[(chunk, self.clean)] = count
        self.attention_calls += 1
        return self.original_attention(query, torch.cat([history_key, key[:, -tokens:]], dim=1),
                                       torch.cat([history_value, value[:, -tokens:]], dim=1), **kwargs)

    def metadata(self):
        denoise = {c: v for (c, clean), v in self.attended.items() if not clean}
        full = sum(c * self.tokens for c in denoise) if self.tokens else 0
        used = sum(denoise.values())
        return {"method": "DeCoPrune port (token-level denoising-consistency KV pruning); diagnostic",
                "spec": "reproduction/decoprune-spec.md", "threshold": self.threshold,
                "score_type": "mse", "generation_step_index": self.step_index, "rope_reindex": self.rope,
                "virtual_span": VIRTUAL_SPAN, "recent_frames": RECENT_FRAMES, "local_window_chunks": 1,
                "chunk0_unscored_kept_whole": True,
                "historical_kv_pruning_ratio": (1 - used / full) if full else None,
                "history_tokens_attended_per_chunk": {str(c): v for c, v in sorted(denoise.items())},
                "chunk_scores": [self.records[c] for c in sorted(self.records)],
                "archive_bytes": sum(e["k"].numel() * e["k"].element_size() + e["v"].numel() * e["v"].element_size()
                                     for layer in self.archive.values() for e in layer.values()),
                "attention_calls_rebuilt": self.attention_calls}

    def close(self):
        self.module.attention = self.original_attention
        self.pipeline._convert_flow_pred_to_x0 = self.original_convert
        for handle in self.handles:
            handle.remove()
