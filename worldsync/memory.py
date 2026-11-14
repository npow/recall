"""A WorldTrace-Landmark-inspired, pre-RoPE memory policy.

ABot already stores pre-RoPE keys and rebases the visible window. We change
which frames enter that window. This is NOT the combined WorldTrace algorithm:
there is no attention-response scoring or field-summary merging here.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class MemoryConfig:
    mode: str = "landmark"
    window_frames: int = 21
    landmark_frames: int = 6
    novelty_threshold: float = 0.15

    def __post_init__(self):
        if self.mode not in {"window", "landmark"}:
            raise ValueError("mode must be window or landmark")
        if not 0 < self.landmark_frames < self.window_frames <= 21:
            raise ValueError("require 0 < landmark_frames < window_frames <= 21 (ABot RoPE limit)")
        if not 0 <= self.novelty_threshold <= 2:
            raise ValueError("novelty_threshold must be in [0, 2]")


@dataclass
class Frame:
    index: int
    key: torch.Tensor
    value: torch.Tensor
    landmark: bool = False


class LandmarkMemory:
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.recent: list[Frame] = []
        self.landmarks: list[Frame] = []
        self.pending: list[Frame] = []
        self.pending_start: int | None = None
        self.previous_descriptor = None
        self.committed_frames = 0

    @staticmethod
    def descriptor(key):
        # Uniform spatial samples, all heads/channels. Independent of RoPE.
        stride = max(1, key.shape[1] // 64)
        return F.normalize(key[:, ::stride].float().flatten(), dim=0)

    def _commit(self):
        for frame in self.pending:
            desc = self.descriptor(frame.key)
            distance = (1 - torch.dot(desc, self.previous_descriptor)).item() if self.previous_descriptor is not None else 2.0
            frame.landmark = distance >= self.config.novelty_threshold
            self.previous_descriptor = desc
            self.recent.append(frame)
            self.committed_frames += 1

    def view(self, key, value, start_frame: int, frame_tokens: int):
        if key.shape != value.shape or key.ndim != 4 or key.shape[0] != 1:
            raise ValueError("expected matching batch-one [B, tokens, heads, dim] K/V")
        if frame_tokens <= 0 or key.shape[1] % frame_tokens:
            raise ValueError("keys must contain complete latent frames")
        count = key.shape[1] // frame_tokens
        recent_budget = self.config.window_frames - self.config.landmark_frames
        if not 0 < count <= recent_budget:
            raise ValueError("current chunk must fit inside the recent-frame budget")
        if self.pending_start is not None and start_frame != self.pending_start:
            if start_frame != self.pending_start + len(self.pending):
                raise ValueError("non-contiguous stream; reset memory before starting another episode")
            self._commit()
        self.pending_start = start_frame
        # Denoising revisits the same chunk several times. Replace pending keys;
        # commit only when the next chunk begins, after the clean-context pass.
        self.pending = [Frame(start_frame + i,
                              key[:, i * frame_tokens:(i + 1) * frame_tokens].detach().clone(),
                              value[:, i * frame_tokens:(i + 1) * frame_tokens].detach().clone())
                        for i in range(count)]
        if self.config.mode == "window":
            self.recent = self.recent[-(self.config.window_frames - count):]
        elif self.committed_frames + count > self.config.window_frames:
            keep = recent_budget - count
            split = len(self.recent) - keep
            evicted, self.recent = self.recent[:split], self.recent[split:]
            self.landmarks.extend(frame for frame in evicted if frame.landmark)
            self.landmarks = self.landmarks[-self.config.landmark_frames:]
        frames = self.landmarks + self.recent + self.pending
        if len(frames) > self.config.window_frames:
            raise RuntimeError("attended frame budget exceeded")
        return torch.cat([f.key for f in frames], dim=1), torch.cat([f.value for f in frames], dim=1)

    def stats(self):
        return {"landmark_ids": [f.index for f in self.landmarks],
                "recent_ids": [f.index for f in self.recent],
                "pending_ids": [f.index for f in self.pending],
                "committed_frames": self.committed_frames}


def memory_view(module, kv_cache, key, value, current_start, frame_tokens):
    config = module._worldsync_config
    # ABot reuses cache dictionaries when reset_stream is called. The first
    # block's repeated denoising calls may reset this empty history harmlessly.
    if current_start == 0 or "_worldsync" not in kv_cache:
        kv_cache["_worldsync"] = LandmarkMemory(config)
    if current_start % frame_tokens:
        raise ValueError("current_start must be a whole-frame token offset")
    return kv_cache["_worldsync"].view(key, value, current_start // frame_tokens, frame_tokens)


def enable_memory(model, config: MemoryConfig):
    """Enable only patched ABot self-attention layers, leaving other attention alone."""
    layers = [m for m in model.modules() if m.__class__.__name__ == "CausalWanSelfAttention"]
    if not layers:
        raise ValueError("no ABot causal self-attention layers found")
    for layer in layers:
        if not layer.use_relative_rope or layer.sink_size:
            raise ValueError("adaptation requires relative RoPE and sink_size=0")
        if layer.local_attn_size != config.window_frames:
            raise ValueError("model local attention size must match the experiment window")
        layer._worldsync_config = config
    return len(layers)
