"""Cross-agent RECALL: one pose-indexed canonical KV archive, private live caches.

For the first GPU diagnostic both cameras rotate about a common origin from the
same conditioning image. Static scene revision 0; no inferred multi-view geometry.
Only fully committed clean frames from another agent, at a matching pose, enter
self-attention. All stored keys are pre-RoPE, rephased to the reader's timeline.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch

from worldsync.lingbot_retrieval import PoseKVMemory


def pose_gap(a, b):
    distance = float(np.linalg.norm(a[:3, 3]-b[:3, 3]))
    cosine = np.clip((np.trace(a[:3, :3].T@b[:3, :3])-1)/2, -1, 1)
    return distance, math.degrees(math.acos(float(cosine)))


@dataclass
class Frame:
    id: str
    actor: str
    source_pass: str
    index: int
    pose: np.ndarray
    revision: int
    geometry: tuple
    kv: torch.Tensor  # [layers, 2, tokens, heads, head_dim], canonical BF16 CPU
    order: int = 0


class SharedPoseArchive:
    def __init__(self, frames_per_actor=8):
        if frames_per_actor < 1:
            raise ValueError("archive needs a positive per-agent frame limit")
        self.limit = frames_per_actor
        self.frames = {}
        self.commits = 0
        self.publications = []

    def commit(self, frames):
        # One clean chunk becomes visible atomically after every layer is captured.
        for frame in frames:
            frame.order = self.commits
            self.commits += 1
            self.frames[frame.id] = frame
            self.publications.append({"id": frame.id, "actor": frame.actor, "source_pass": frame.source_pass,
                                      "index": frame.index, "revision": frame.revision,
                                      "pose": frame.pose.tolist(), "layers": frame.kv.shape[0]})
        for actor in {frame.actor for frame in frames}:
            while len(pool := [f for f in self.frames.values() if f.actor == actor]) > self.limit:
                # Remove the newest redundant pose, preserving early visual evidence.
                def redundancy(frame):
                    nearest = min(d/4+a/60 for other in pool if other.id != frame.id
                                  for d, a in [pose_gap(frame.pose, other.pose)])
                    return nearest, -frame.order
                victim = min(pool, key=redundancy)
                del self.frames[victim.id]

    def select(self, actor, target, revision=0, count=4, geometry=None,
               max_angle=20, max_distance=.01):
        candidates = []
        for frame in self.frames.values():
            if frame.actor == actor or frame.revision != revision:
                continue
            if geometry is not None and frame.geometry != geometry:
                continue
            distance, angle = pose_gap(target, frame.pose)
            if distance <= max_distance and angle <= max_angle:
                candidates.append((distance/4+angle/60, frame.order, frame))
        return [f for _, _, f in sorted(candidates, key=lambda r: (r[0], r[1]))[:count]]

    def metadata(self):
        return {"frames": len(self.frames), "frames_per_actor_limit": self.limit,
                "bytes": sum(f.kv.numel()*f.kv.element_size() for f in self.frames.values()),
                "committed_frames": self.commits, "publications": self.publications}


def splice_peer_kv(key, value, peer_key, peer_value, frame_tokens, window=18, sink=6):
    """Keep current queries, early anchors and recent frames within the native budget."""
    current = key.shape[1]//frame_tokens
    retrieved = peer_key.shape[1]//frame_tokens
    keep = min(current, window-retrieved)
    if keep < 4:
        raise ValueError("peer memory would evict the current four latent frames")
    head = sink if current > keep else 0
    tail = keep-head
    if tail < 4:
        raise ValueError("peer memory would remove current query frames")
    k = torch.cat([key[:, :head*frame_tokens], peer_key, key[:, -tail*frame_tokens:]], dim=1)
    v = torch.cat([value[:, :head*frame_tokens], peer_value, value[:, -tail*frame_tokens:]], dim=1)
    return k, v


class SharedPoseKVMemory(PoseKVMemory):
    release_before_decode = True

    def __init__(self, pipeline, archive, actor, source_pass, revision=0, retrieved=4):
        super().__init__(pipeline, retrieved_frames=4, key_positions="virtual")
        self.archive, self.actor, self.source_pass = archive, actor, source_pass
        self.revision, self.count = revision, retrieved
        self.layers = len(pipeline.model.blocks)
        self.pending = {}
        self.captured_layers = set()
        self.geometry = None
        self.injections = {}

    def begin(self, chunk, poses):
        if self.pending:
            raise RuntimeError("previous clean chunk was not committed")
        self.chunk = chunk
        self.poses = np.asarray(poses)
        self.cached.clear()
        self.selected = self.archive.select(self.actor, self.poses[-1], self.revision,
                                            self.count, self.geometry)
        self.trace.append({"chunk": chunk, "reader": self.actor, "revision": self.revision,
                           "target_pose": self.poses[-1].tolist(),
                           "peer_frames": [{"id": f.id, "actor": f.actor, "source_pass": f.source_pass,
                                            "frame": f.index, "pose_gap": pose_gap(self.poses[-1], f.pose)}
                                           for f in self.selected],
                           "attention_replaced": False, "layers_injected": []})

    def attend(self, query, key, value, **kwargs):
        if self.active is None:
            return self.original_attention(query, key, value, **kwargs)
        layer, start, tokens, grid, freqs = self.active
        geometry = (int(grid[0, 1]), int(grid[0, 2]), tokens, query.shape[2], query.shape[3])
        self.geometry = geometry
        if self.clean:
            if self.raw_key is None:
                raise RuntimeError("shared archive needs canonical pre-RoPE keys")
            raw = self.raw_key.view_as(query)
            current_value = value[:, -query.shape[1]:]
            for offset in range(query.shape[1]//tokens):
                index = start//tokens+offset
                if index < 6:
                    continue  # same reference anchors are not evidence of a peer visit
                if index not in self.pending:
                    data = torch.empty((self.layers, 2, tokens, query.shape[2], query.shape[3]),
                                       dtype=query.dtype, device="cpu")
                    self.pending[index] = Frame(f"{self.actor}:{self.source_pass}:{index}", self.actor,
                        self.source_pass, index, self.poses[index].copy(), self.revision, geometry, data)
                lo, hi = offset*tokens, (offset+1)*tokens
                self.pending[index].kv[layer, 0].copy_(raw[0, lo:hi].detach())
                self.pending[index].kv[layer, 1].copy_(current_value[0, lo:hi].detach())
            self.captured_layers.add(layer)
        selected = [f for f in self.selected if f.geometry == geometry]
        # Use peer memory once the native attention window is full, as in RECALL.
        if key.shape[1] != 18*tokens or not selected:
            return self.original_attention(query, key, value, **kwargs)
        if layer not in self.cached:
            canonical = torch.cat([f.kv[layer, 0] for f in selected], dim=0)[None].to(key.device)
            peer_value = torch.cat([f.kv[layer, 1] for f in selected], dim=0)[None].to(value.device)
            virtual_grid = grid.clone()
            virtual_grid[:, 0] = len(selected)
            tail = 18-6-len(selected)
            with torch.amp.autocast("cuda", enabled=False):
                peer_key = self.module.causal_rope_apply(canonical, virtual_grid, freqs,
                    start_frame=max(0, start//tokens-tail-len(selected))).to(key.dtype)
            self.cached[layer] = peer_key, peer_value
        peer_key, peer_value = self.cached[layer]
        k, v = splice_peer_kv(key, value, peer_key, peer_value, tokens)
        self.trace[-1]["attention_replaced"] = True
        if layer not in self.trace[-1]["layers_injected"]:
            self.trace[-1]["layers_injected"].append(layer)
        self.injections[layer] = self.injections.get(layer, 0)+1
        self.retrieval_calls += 1
        return self.original_attention(query, k, v, **kwargs)

    def observe(self, chunk, pixels):
        if self.pending:
            if self.captured_layers != set(range(self.layers)):
                raise RuntimeError("incomplete clean-layer capture; refusing shared publication")
            self.archive.commit(list(self.pending.values()))
        self.pending.clear()
        self.captured_layers.clear()

    def metadata(self):
        return {"method": "cross-agent pose-matched RECALL, canonical visual KV injected into self-attention",
                "actor": self.actor, "source_pass": self.source_pass, "revision": self.revision,
                "attended_latent_frames": 18, "peer_frame_limit": self.count,
                "key_positions": "canonical pre-RoPE keys rephased to reader virtual history",
                "pose_gate": {"max_angle_degrees": 20, "max_distance": .01},
                "scope": "static scene; same start image, common origin; rotation-only paths",
                "injections_by_layer": self.injections, "retrieval_attention_calls": self.retrieval_calls,
                "trace": self.trace, "shared_archive": self.archive.metadata()}
