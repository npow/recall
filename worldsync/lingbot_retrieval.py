"""Bounded early-view KV retrieval diagnostic, inspired by WorldKV/WorldTrace.

Not a reproduction of either paper: archive 16 early non-sink latent frames,
retrieve four by commanded pose, and place canonical keys at recent virtual
times. Native sink/recent/current slots retain their original keys and values.
"""
import math
import numpy as np
import torch


def select_frames(poses, available, current_start, count=4):
    target = poses[-1]
    ranked = []
    for index in available:
        if index < 6 or index >= current_start - 4:
            continue  # Never duplicate sink, recent, current or future frames.
        pose = poses[index]
        distance = float(np.linalg.norm(target[:3, 3] - pose[:3, 3]))
        angle = math.acos(float(np.clip((np.trace(target[:3, :3].T @ pose[:3, :3]) - 1) / 2, -1, 1)))
        if distance <= 8 and angle <= math.pi / 3:
            ranked.append((distance / 4 + angle / (math.pi / 3), index))
    return [index for _, index in sorted(ranked)[:count]]


class PoseKVMemory:
    def __init__(self, pipeline, key_positions="virtual"):
        if key_positions not in {"virtual", "native"}:
            raise ValueError("key_positions must be virtual or native")
        self.key_positions = key_positions
        from wan.modules import model_fast
        self.module = model_fast
        self.original_attention = model_fast.attention
        self.banks = {}
        self.cached = {}
        self.available = set()
        self.handles = []
        self.active = None
        self.raw_key = None
        self.clean = False
        self.selected = []
        self.poses = None
        self.chunk = -1
        self.trace = []
        self.frame_tokens = None
        self.retrieval_calls = 0
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
        model_fast.attention = self.attend

    def begin(self, chunk, poses):
        self.chunk = chunk
        self.poses = np.asarray(poses)
        self.cached.clear()
        self.selected = select_frames(self.poses, self.available, chunk * 4)
        self.trace.append({"chunk": chunk, "retrieved_frame_ids": self.selected,
                           "archive_frame_ids": sorted(self.available), "attention_replaced": False})

    def attend(self, query, key, value, **kwargs):
        if self.active is None:
            return self.original_attention(query, key, value, **kwargs)
        layer, start, frame_tokens, grid, freqs = self.active
        self.frame_tokens = frame_tokens
        first_frame = start // frame_tokens
        bank = self.banks.setdefault(layer, {})
        if self.clean:
            if self.raw_key is None:
                raise RuntimeError("canonical key capture missing")
            raw = self.raw_key.view_as(query) if self.key_positions == "virtual" else key[:, -query.shape[1]:]
            # Keep a bounded, explicit initial history. No hidden unbounded bank.
            for offset in range(query.shape[1] // frame_tokens):
                index = first_frame + offset
                if 6 <= index < 22:
                    lo, hi = offset * frame_tokens, (offset + 1) * frame_tokens
                    v_current = value[:, -query.shape[1]:]
                    bank[index] = (raw[:, lo:hi].detach().to("cpu", copy=True),
                                   v_current[:, lo:hi].detach().to("cpu", copy=True))
                    if layer == 0:
                        self.available.add(index)
        # Same 18-frame attended budget as stock: sink6 + retrieved4 + tail8.
        if len(self.selected) != 4 or key.shape[1] != 18 * frame_tokens:
            return self.original_attention(query, key, value, **kwargs)
        if layer not in self.cached:
            if any(index not in bank for index in self.selected):
                raise RuntimeError("selected memory not populated in every layer")
            canonical = torch.cat([bank[index][0] for index in self.selected], dim=1).to(key.device)
            retrieved_value = torch.cat([bank[index][1] for index in self.selected], dim=1).to(value.device)
            virtual_grid = grid.clone()
            virtual_grid[:, 0] = 4
            if self.key_positions == "native":
                retrieved_key = canonical  # Exact original post-RoPE keys, no positional reassignment.
            else:
                with torch.amp.autocast("cuda", enabled=False):
                    retrieved_key = self.module.causal_rope_apply(canonical, virtual_grid, freqs,
                        start_frame=first_frame - 8).to(key.dtype)
            self.cached[layer] = (retrieved_key, retrieved_value)
        retrieved_key, retrieved_value = self.cached[layer]
        combined_key = torch.cat([key[:, :6 * frame_tokens], retrieved_key, key[:, -8 * frame_tokens:]], dim=1)
        combined_value = torch.cat([value[:, :6 * frame_tokens], retrieved_value, value[:, -8 * frame_tokens:]], dim=1)
        if combined_key.shape != key.shape:
            raise RuntimeError("retrieval changed attended token budget")
        self.trace[-1]["attention_replaced"] = True
        self.retrieval_calls += 1
        return self.original_attention(query, combined_key, combined_value, **kwargs)

    def metadata(self):
        return {"method": "bounded early-view pose KV retrieval; diagnostic, not paper reproduction",
                "archive_limit_latent_frames": 16, "retrieved_latent_frames": 4,
                "attended_latent_frames": 18, "sink_latent_frames": 6, "tail_latent_frames": 8,
                "pose_source": "commanded poses, not reconstructed realized poses",
                "key_positions": self.key_positions,
                "key_storage": ("exact native post-RoPE keys; original temporal/spatial positions retained" if self.key_positions == "native"
                                else "canonical pre-RoPE BF16 keys; reapply spatial and virtual temporal RoPE on read"),
                "archive_bytes": sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                                     for bank in self.banks.values() for k, v in bank.values()),
                "retrieval_attention_calls": self.retrieval_calls, "trace": self.trace}

    def close(self):
        self.module.attention = self.original_attention
        for handle in self.handles:
            handle.remove()
