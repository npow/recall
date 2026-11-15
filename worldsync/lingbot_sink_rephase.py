"""Pose-gated initial-view key rephasing, retaining the native attention budget.

This isolates positional staleness from missing visual content. The native sink
already retains initial values; on a commanded revisit only its temporal key
positions move beside the rolling history. Native persistent caches are unedited.
"""
import math
import numpy as np
import torch

from worldsync.lingbot_retrieval import PoseKVMemory


class SinkRephaseMemory(PoseKVMemory):
    def begin(self, chunk, poses):
        self.chunk = chunk
        self.poses = np.asarray(poses)
        self.cached.clear()
        initial, target = self.poses[0], self.poses[-1]
        distance = float(np.linalg.norm(initial[:3, 3] - target[:3, 3]))
        angle = math.acos(float(np.clip((np.trace(initial[:3, :3].T @ target[:3, :3]) - 1) / 2, -1, 1)))
        self.overlap = chunk * 4 >= 18 and distance <= 2 and angle <= math.pi / 4
        self.trace.append({"chunk": chunk, "distance_from_initial": distance,
                           "angle_from_initial_degrees": math.degrees(angle),
                           "pose_overlap": self.overlap, "attention_replaced": False})

    def attend(self, query, key, value, **kwargs):
        if self.active is None:
            return self.original_attention(query, key, value, **kwargs)
        layer, start, frame_tokens, grid, freqs = self.active
        first_frame = start // frame_tokens
        bank = self.banks.setdefault(layer, {})
        if self.clean:
            if self.raw_key is None:
                raise RuntimeError("canonical sink key capture missing")
            canonical = self.raw_key.view_as(query)
            for offset in range(query.shape[1] // frame_tokens):
                index = first_frame + offset
                if index < 6:
                    bank[index] = canonical[:, offset * frame_tokens:(offset + 1) * frame_tokens].detach().to("cpu", copy=True)
        if not self.overlap or key.shape[1] != 18 * frame_tokens:
            return self.original_attention(query, key, value, **kwargs)
        if layer not in self.cached:
            if set(bank) != set(range(6)):
                raise RuntimeError("initial sink keys are incomplete")
            canonical = torch.cat([bank[index] for index in range(6)], dim=1).to(key.device)
            virtual_grid = grid.clone()
            virtual_grid[:, 0] = 6
            # The native rolling tail has twelve frames, ending at current+3.
            # Place six sink frames immediately before it: current-14..current-9.
            with torch.amp.autocast("cuda", enabled=False):
                self.cached[layer] = self.module.causal_rope_apply(canonical, virtual_grid, freqs,
                    start_frame=first_frame - 14).to(key.dtype)
        combined = torch.cat([self.cached[layer], key[:, 6 * frame_tokens:]], dim=1)
        if combined.shape != key.shape:
            raise RuntimeError("sink rephasing changed the attended budget")
        self.trace[-1]["attention_replaced"] = True
        self.trace[-1]["virtual_sink_start"] = first_frame - 14
        self.retrieval_calls += 1
        return self.original_attention(query, combined, value, **kwargs)

    def metadata(self):
        return {"method": "pose-gated canonical sink key temporal rephasing v1; not paper reproduction",
                "attended_latent_frames": 18, "sink_latent_frames": 6, "tail_latent_frames": 12,
                "archive_limit_latent_frames": 6, "value_storage": "native sink values, unchanged",
                "persistent_cache_edited": False,
                "pose_source": "commanded poses, not reconstructed realized poses",
                "trigger": "within2 movement units and45 degrees of initial pose, after eviction begins",
                "archive_bytes": sum(key.numel() * key.element_size() for bank in self.banks.values() for key in bank.values()),
                "retrieval_attention_calls": self.retrieval_calls, "trace": self.trace}
