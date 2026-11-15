"""Bounded early-view KV retrieval diagnostic, inspired by WorldKV/WorldTrace.

Not a reproduction of either paper: archive 16 early non-sink latent frames,
retrieve four by commanded pose, and place canonical keys at recent virtual
times. Native sink/recent/current slots retain their original keys and values.
"""
import math
import numpy as np
import torch


def select_frames(poses, available, current_start, count=4, diverse=False, recent_frames=4):
    target = poses[-1]
    ranked = []
    for index in available:
        if index < 6 or index >= current_start - recent_frames:
            continue  # Never duplicate sink, recent, current or future frames.
        pose = poses[index]
        distance = float(np.linalg.norm(target[:3, 3] - pose[:3, 3]))
        angle = math.acos(float(np.clip((np.trace(target[:3, :3].T @ pose[:3, :3]) - 1) / 2, -1, 1)))
        if distance <= 8 and angle <= math.pi / 3:
            ranked.append((distance / 4 + angle / (math.pi / 3), index))
    ordered = [index for _, index in sorted(ranked)]
    if not diverse:
        return ordered[:count]
    selected = []
    for index in ordered:
        independent = True
        for old in selected:
            distance = float(np.linalg.norm(poses[index][:3, 3] - poses[old][:3, 3]))
            angle = math.acos(float(np.clip((np.trace(poses[index][:3, :3].T @ poses[old][:3, :3]) - 1) / 2, -1, 1)))
            if distance < 2 and angle < math.pi / 12 - 1e-6:
                independent = False
                break
        if independent:
            selected.append(index)
            if len(selected) == count:
                break
    return selected


class PoseKVMemory:
    def __init__(self, pipeline, key_positions="virtual", retrieved_frames=4, selection="nearest", cache_frames=18, history_weight=None):
        if retrieved_frames not in {4, 8} or selection not in {"nearest", "diverse"}:
            raise ValueError("retrieval supports4/8 frames and nearest/diverse selection")
        if cache_frames not in {18, 22}:
            raise ValueError("retrieval supports18/22 attended latent frames")
        if history_weight is not None and not 0 <= history_weight <= 1:
            raise ValueError("history output weight must be in[0,1]")
        self.history_weight = history_weight
        self.cache_frames = cache_frames
        self.retrieved_frames = retrieved_frames
        self.selection = selection
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
        self.archive_range = (6, 22)  # latent frames archived; subclasses may widen it
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
        self.selected = select_frames(self.poses, self.available, chunk * 4, count=self.retrieved_frames, diverse=self.selection == "diverse",
                                      recent_frames=self.cache_frames - 14)
        self.trace.append({"chunk": chunk, "retrieved_frame_ids": self.selected,
                           "archive_frame_ids": sorted(self.available), "attention_replaced": False})

    def release_read_cache(self):
        """Drop temporary GPU reads after the clean pass, before VAE decoding.

        The CPU archive and the model's persistent native KV cache are untouched.
        Reads are already rebuilt in begin() for the next chunk.
        """
        if self.active is not None:
            raise RuntimeError("cannot release retrieval reads during attention")
        released = sum(tensor.numel() * tensor.element_size()
                       for pair in self.cached.values() for tensor in pair)
        self.cached.clear()
        if self.trace:
            self.trace[-1]["temporary_read_bytes_released_before_decode"] = released
        return released

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
                if self.archive_range[0] <= index < self.archive_range[1]:
                    lo, hi = offset * frame_tokens, (offset + 1) * frame_tokens
                    v_current = value[:, -query.shape[1]:]
                    bank[index] = (raw[:, lo:hi].detach().to("cpu", copy=True),
                                   v_current[:, lo:hi].detach().to("cpu", copy=True))
                    if layer == 0:
                        self.available.add(index)
        # Preserve the declared attention budget; keep at least four current frames.
        if len(self.selected) < 4 or key.shape[1] != self.cache_frames * frame_tokens:
            return self.original_attention(query, key, value, **kwargs)
        if layer not in self.cached:
            if any(index not in bank for index in self.selected):
                raise RuntimeError("selected memory not populated in every layer")
            canonical = torch.cat([bank[index][0] for index in self.selected], dim=1).to(key.device)
            retrieved_value = torch.cat([bank[index][1] for index in self.selected], dim=1).to(value.device)
            virtual_grid = grid.clone()
            virtual_grid[:, 0] = len(self.selected)
            if self.key_positions == "native":
                retrieved_key = canonical  # Exact original post-RoPE keys, no positional reassignment.
            else:
                with torch.amp.autocast("cuda", enabled=False):
                    retrieved_key = self.module.causal_rope_apply(canonical, virtual_grid, freqs,
                        start_frame=first_frame + 10 - self.cache_frames).to(key.dtype)
            self.cached[layer] = (retrieved_key, retrieved_value)
        retrieved_key, retrieved_value = self.cached[layer]
        tail = self.cache_frames - 6 - len(self.selected)
        if (6 + tail) * frame_tokens + retrieved_key.shape[1] != key.shape[1]:
            raise RuntimeError("retrieval changed attended token budget")
        self.trace[-1]["attention_replaced"] = True
        self.trace[-1]["retained_tail_latent_frames"] = tail
        self.retrieval_calls += 1
        if self.history_weight is not None:
            # Keep fused attention in both branches. An arbitrary dense logit
            # mask would disable the current fused SDPA path and exhaust VRAM.
            recent_key = torch.cat([key[:, :6 * frame_tokens], key[:, -tail * frame_tokens:]], dim=1)
            recent_value = torch.cat([value[:, :6 * frame_tokens], value[:, -tail * frame_tokens:]], dim=1)
            recent_output = self.original_attention(query, recent_key, recent_value, **kwargs)
            history_output = self.original_attention(query, retrieved_key, retrieved_value, **kwargs)
            return torch.lerp(recent_output, history_output, self.history_weight)
        combined_key = torch.cat([key[:, :6 * frame_tokens], retrieved_key, key[:, -tail * frame_tokens:]], dim=1)
        combined_value = torch.cat([value[:, :6 * frame_tokens], retrieved_value, value[:, -tail * frame_tokens:]], dim=1)
        return self.original_attention(query, combined_key, combined_value, **kwargs)

    def metadata(self):
        return {"method": "bounded early-view pose KV retrieval; diagnostic, not paper reproduction",
                "archive_limit_latent_frames": 16, "retrieved_latent_frames": self.retrieved_frames,
                "retrieved_frame_count_is_limit": True, "selection": self.selection,
                "attention_composition": "joint softmax" if self.history_weight is None else "separately normalized recent/history outputs",
                "history_output_weight": self.history_weight,
                "attention_branches_per_retrieval": 1 if self.history_weight is None else 2,
                "attended_latent_frames": self.cache_frames, "sink_latent_frames": 6,
                "tail_latent_frames": self.cache_frames - 6 - self.retrieved_frames,
                "excluded_recent_archive_frames": self.cache_frames - 14,
                "tail_frame_count_is_minimum": True,
                "temporary_gpu_read_lifetime": "denoising and clean pass only; released before VAE decode",
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
