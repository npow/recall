"""Bounded pose-indexed KV memory for LingBot-World2 (training-free diagnostic).

Replaces the audited-buggy WorldKV-style relevance retrieval:
* selection uses commanded camera pose (position + heading) against the chunk's
  target pose, not a content-blind query captured at the noisiest denoising step;
* the gate derives from the real stock window (sink + 12 frames), so other
  window/sink settings cannot silently run stock;
* the archive is bounded: at most `archive_frames` latent frames, evicted by pose
  redundancy (the frame whose pose is closest to another archived frame goes first),
  stored as canonical pre-RoPE bf16 K/V on CPU (about 1.3 GB per latent frame).

Attended window = sink + K retrieved + tail(8): K above 4 enlarges the window and
GPU memory (about 1.25 GB per extra latent frame on an H200).
"""
import math

import numpy as np
import torch

from worldsync.lingbot_retrieval import PoseKVMemory

TAIL = 8
STOCK_RECENT = 12


def pose_cost(target, pose):
    distance = float(np.linalg.norm(target[:3, 3] - pose[:3, 3]))
    cos = np.clip((np.trace(target[:3, :3].T @ pose[:3, :3]) - 1) / 2, -1, 1)
    return distance / 4 + math.acos(float(cos)) / (math.pi / 3)


def select_by_pose(poses, available, first_frame, sink, count):
    """Pick the `count` archived frames whose pose is closest to the chunk's target pose."""
    target = poses[-1]
    candidates = [i for i in available if sink <= i < first_frame - 4 and i < len(poses)]
    return sorted(sorted(candidates, key=lambda i: pose_cost(target, poses[i]))[:count])


def evict_redundant(poses, available, limit, protect_from):
    """Drop the archived frame whose pose is closest to a neighbour until within `limit`."""
    keep = set(available)
    while len(keep) > limit:
        pool = sorted(i for i in keep if i < protect_from and i < len(poses))
        if len(pool) < 2:
            break
        best, victim = None, None
        for position, i in enumerate(pool):
            nearest = min(pose_cost(poses[i], poses[j]) for j in pool if j != i)
            if best is None or nearest < best or (nearest == best and i < victim):
                best, victim = nearest, i
        keep.discard(victim)
    return keep


class BudgetPoseMemory(PoseKVMemory):
    def __init__(self, pipeline, sink_size=6, local_attn_size=18, retrieved=10, archive_frames=24, rope="virtual"):
        super().__init__(pipeline)
        if rope not in ("virtual", "native"):
            raise ValueError("rope must be virtual or native")
        self.sink = sink_size
        self.window = local_attn_size
        self.retrieved = retrieved
        self.archive_limit = archive_frames
        self.rope = rope
        self.latest_start = 0

    def begin(self, chunk, poses):
        self.chunk = chunk
        self.poses = np.asarray(poses)
        self.cached.clear()
        first_frame = chunk * 4
        self.latest_start = first_frame
        # Eviction is decided from poses only, once per chunk, identically for every layer.
        keep = evict_redundant(self.poses, self.available, self.archive_limit, first_frame - 4)
        self.available = set(keep)
        for bank in self.banks.values():
            for index in [i for i in bank if i not in keep]:
                del bank[index]
        self.selected = select_by_pose(self.poses, self.available, first_frame, self.sink, self.retrieved)
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
            raw = self.raw_key.view_as(query)
            v_current = value[:, -query.shape[1]:]
            for offset in range(query.shape[1] // frame_tokens):
                index = first_frame + offset
                if index < self.sink:
                    continue
                lo, hi = offset * frame_tokens, (offset + 1) * frame_tokens
                bank[index] = (raw[:, lo:hi].detach().to("cpu", copy=True),
                               v_current[:, lo:hi].detach().to("cpu", copy=True))
                if layer == 0:
                    self.available.add(index)
        stock_frames = self.window
        if key.shape[1] % frame_tokens or key.shape[1] != stock_frames * frame_tokens or not self.selected:
            return self.original_attention(query, key, value, **kwargs)
        if layer not in self.cached:
            missing = [i for i in self.selected if i not in bank]
            if missing:
                raise RuntimeError(f"selected memory frames {missing} not archived in layer {layer}")
            canonical = torch.cat([bank[i][0] for i in self.selected], dim=1).to(key.device)
            retrieved_value = torch.cat([bank[i][1] for i in self.selected], dim=1).to(value.device)
            with torch.amp.autocast("cuda", enabled=False):
                if self.rope == "virtual":
                    virtual_grid = grid.clone()
                    virtual_grid[:, 0] = len(self.selected)
                    retrieved_key = self.module.causal_rope_apply(canonical, virtual_grid, freqs,
                        start_frame=max(first_frame - TAIL - len(self.selected), 0))
                else:
                    single = grid.clone()
                    single[:, 0] = 1
                    parts = [self.module.causal_rope_apply(canonical[:, p * frame_tokens:(p + 1) * frame_tokens],
                             single, freqs, start_frame=i) for p, i in enumerate(self.selected)]
                    retrieved_key = torch.cat(parts, dim=1)
            self.cached[layer] = (retrieved_key.to(key.dtype), retrieved_value)
        retrieved_key, retrieved_value = self.cached[layer]
        combined_key = torch.cat([key[:, :self.sink * frame_tokens], retrieved_key, key[:, -TAIL * frame_tokens:]], dim=1)
        combined_value = torch.cat([value[:, :self.sink * frame_tokens], retrieved_value, value[:, -TAIL * frame_tokens:]], dim=1)
        self.trace[-1]["attention_replaced"] = True
        self.retrieval_calls += 1
        return self.original_attention(query, combined_key, combined_value, **kwargs)

    def metadata(self):
        data = super().metadata()
        data.update({"method": "bounded pose-indexed KV memory; diagnostic, not paper reproduction",
                     "retrieved_latent_frames": self.retrieved, "archive_limit_latent_frames": self.archive_limit,
                     "attended_latent_frames": self.sink + self.retrieved + TAIL, "rope": self.rope,
                     "selection": "commanded pose distance + heading to the chunk's target pose; eviction by pose redundancy"})
        return data


class StoredBudgetMemory(BudgetPoseMemory):
    """Bounded pose memory with a storage codec and an optional layer subset (see lingbot_memstore).

    Same selection, eviction, window and RoPE as BudgetPoseMemory. Differences: archived K/V go through
    `codec`; only `layers` archive and substitute (other layers attend the stock window); retrieved frames
    stay in stored form (GPU if small, else pinned CPU) and are decoded per attention call.
    """
    release_before_decode = True

    def __init__(self, pipeline, codec, layers=None, gpu_resident_bytes=7.5e9, **kwargs):
        super().__init__(pipeline, **kwargs)
        from worldsync import lingbot_memstore
        self.store = lingbot_memstore
        self.codec = codec
        self.layers = None if layers is None else set(layers)
        self.gpu_resident_bytes = gpu_resident_bytes
        self.frame_bytes = {}
        self.resident = None
        self.decode_seconds = 0.0

    def begin(self, chunk, poses):
        super().begin(chunk, poses)
        self.resident = None

    def release_read_cache(self):
        released = sum(self.store.nbytes(entry) for entries in self.cached.values() for entry in entries)
        self.cached.clear()
        if self.trace:
            self.trace[-1]["temporary_read_bytes_released_before_decode"] = released
        return released

    def _uses(self, layer):
        return self.layers is None or layer in self.layers

    def attend(self, query, key, value, **kwargs):
        if self.active is None:
            return self.original_attention(query, key, value, **kwargs)
        import time
        layer, start, frame_tokens, grid, freqs = self.active
        self.frame_tokens = frame_tokens
        first_frame = start // frame_tokens
        uses = self._uses(layer)
        bank = self.banks.setdefault(layer, {})
        if self.clean:
            if self.raw_key is None:
                raise RuntimeError("canonical key capture missing")
            raw = self.raw_key.view_as(query)
            v_current = value[:, -query.shape[1]:]
            hw = (int(grid[0, 1]), int(grid[0, 2]))
            for offset in range(query.shape[1] // frame_tokens):
                index = first_frame + offset
                if index < self.sink:
                    continue
                if layer == 0:
                    self.available.add(index)
                if uses:
                    lo, hi = offset * frame_tokens, (offset + 1) * frame_tokens
                    bank[index] = self.codec.encode(raw[:, lo:hi], v_current[:, lo:hi], hw)
                    self.frame_bytes[layer] = self.store.nbytes(bank[index])
        if (not uses or key.shape[1] % frame_tokens or key.shape[1] != self.window * frame_tokens
                or not self.selected):
            return self.original_attention(query, key, value, **kwargs)
        started = time.monotonic()
        if layer not in self.cached:
            missing = [i for i in self.selected if i not in bank]
            if missing:
                raise RuntimeError(f"selected memory frames {missing} not archived in layer {layer}")
            if self.resident is None:
                used = len(self.layers) if self.layers is not None else self.layer_count
                total = sum(self.store.nbytes(bank[i]) for i in self.selected) * used
                self.resident = "gpu" if total <= self.gpu_resident_bytes else "cpu"
                self.trace[-1]["read_resident"] = self.resident
                self.trace[-1]["read_stored_bytes_all_layers"] = total
            device = key.device if self.resident == "gpu" else "cpu"
            self.cached[layer] = [self.store.to_device(bank[i], device, pin=True) for i in self.selected]
        keys, values = [], []
        virtual_start = max(first_frame - TAIL - len(self.selected), 0)
        with torch.amp.autocast("cuda", enabled=False):
            for position, (frame, entry) in enumerate(zip(self.selected, self.cached[layer])):
                if self.resident == "cpu":
                    entry = self.store.to_device(entry, key.device)
                canonical, retrieved_value, index = self.codec.decode(entry, key.dtype)
                where = frame if self.rope == "native" else virtual_start + position
                keys.append(self.store.positioned_keys(self.module.causal_rope_apply, canonical, index,
                                                       frame_tokens, where, grid, freqs).to(key.dtype))
                values.append(retrieved_value.to(value.dtype))
        retrieved_key, retrieved_value = torch.cat(keys, dim=1), torch.cat(values, dim=1)
        combined_key = torch.cat([key[:, :self.sink * frame_tokens], retrieved_key, key[:, -TAIL * frame_tokens:]], dim=1)
        combined_value = torch.cat([value[:, :self.sink * frame_tokens], retrieved_value, value[:, -TAIL * frame_tokens:]], dim=1)
        self.decode_seconds += time.monotonic() - started
        self.trace[-1]["attention_replaced"] = True
        self.retrieval_calls += 1
        return self.original_attention(query, combined_key, combined_value, **kwargs)

    @property
    def layer_count(self):
        return len(self.handles) // 3

    def metadata(self):
        banks, self.banks = self.banks, {}  # the parent sums (k, v) pairs; stored entries are dicts
        try:
            data = super().metadata()
        finally:
            self.banks = banks
        data["archive_bytes"] = sum(self.store.nbytes(entry) for bank in banks.values() for entry in bank.values())
        per_frame = sum(self.frame_bytes.values())
        exact = self.store.EXACT_BYTES_PER_LAYER * self.layer_count
        data.update({"storage_codec": self.codec.name,
                     "storage_layers": "all" if self.layers is None else sorted(self.layers),
                     "stored_bytes_per_archived_frame": per_frame,
                     "exact_bytes_per_archived_frame": exact,
                     "compression_ratio": (exact / per_frame) if per_frame else None,
                     "read_decode_seconds_total": self.decode_seconds})
        return data
