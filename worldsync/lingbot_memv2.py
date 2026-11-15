"""Memory v2: unlimited exact-KV archive on disk + content-verified retrieval.

Fixes the two flaws the full-run audit found in `lingbot_budgetmem.py`
(reproduction/full-run-memory-audit.md):

1. The 24-frame CPU archive was a ~6 s FIFO on moving paths and kept the newer,
   drifted copy of a repeated view. v2 `disk` writes every latent frame's exact
   canonical per-layer K/V (bf16, as before) to local disk as it is produced
   (background writer; about 1.28 GB per latent frame; the directory is deleted
   when the episode ends) and keeps only descriptors, poses and file names in RAM.
   Nothing is evicted, so the first-seen copy of every view is always available.

2. Retrieval trusted commanded poses. Rotation is realized (~0.9x); translation is
   not visually undone, so after W...S the commanded pose says "back at start" while
   the camera is elsewhere and the old method injected the wrong view. v2 offers
   * `verified` (default): pose is the index (one chunk is a 60 degree turn, so the
     current view alone cannot say which stored view comes next), but a stored frame
     is trusted only if nothing but rotation/wait was commanded since it was stored,
     or the place is verified by content (the current view matches a stored view
     taken at the commanded current pose, or matches the candidate itself);
   * `content`: top-K stored frames by descriptor similarity to the current view;
   * `pose`: the old rule (K nearest commanded poses over a 24-frame
     pose-redundancy archive), kept as the control arm.
   `verified` and `content` retrieve nothing when no stored frame qualifies, and the
   untouched stock attention then runs.

Retrieved K/V stay in host RAM and are streamed to the GPU per attention call, so
GPU peak stays close to stock (the old method held K x 1.28 GB on the GPU).
Descriptors come from the decoded frames the rollout already produces, one per
latent frame: DINOv2-small CLS (default) or a pixel thumbnail (no extra model).
"""
import os
import queue
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch

from worldsync.lingbot_budgetmem import BudgetPoseMemory, evict_redundant, pose_cost, select_by_pose

TAIL = 8
STOCK_RECENT = 12


def latent_pixel_indices(chunk, frames):
    """Pixel-frame index (last of its group) for each of the chunk's four latent frames."""
    if chunk == 0:
        if frames != 13:
            raise ValueError("first chunk must decode to 13 frames")
        return [0, 4, 8, 12]
    if frames != 16:
        raise ValueError("later chunks must decode to 16 frames")
    return [3, 7, 11, 15]


def thumbnail_descriptor(image):
    """Model-free view descriptor: 12x20 RGB thumbnail, mean-centred and L2-normalised."""
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].float() / 255
    small = torch.nn.functional.adaptive_avg_pool2d(tensor, (12, 20)).flatten()
    small = small - small.mean()
    return (small / small.norm().clamp_min(1e-6)).numpy()


class DinoDescriptor:
    def __init__(self, device=None, name="facebook/dinov2-small"):
        from transformers import AutoImageProcessor, AutoModel
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, images):
        inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
        with torch.amp.autocast("cuda", enabled=False):
            features = self.model(**inputs).last_hidden_state[:, 0].float()
        return torch.nn.functional.normalize(features, dim=-1).cpu().numpy()


def translation_between(poses, start, end):
    """Commanded translation path length over latent frames (start, end]."""
    positions = np.asarray(poses)[start:end + 1, :3, 3]
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()) if len(positions) > 1 else 0.0


def pose_close(target, pose, max_angle_deg, max_distance):
    distance = float(np.linalg.norm(target[:3, 3] - pose[:3, 3]))
    cos = np.clip((np.trace(target[:3, :3].T @ pose[:3, :3]) - 1) / 2, -1, 1)
    return distance <= max_distance and np.degrees(np.arccos(float(cos))) <= max_angle_deg + 1e-6


def select_verified(poses, descriptors, available, first_frame, sink, count, *, max_angle_deg=60.0, max_distance=2.0,
                    verify_cost=0.35, tau_verify=0.75, tau_direct=0.80):
    """Pose-indexed, content-verified choice of stored latent frames; earliest copies first.

    Returns (selected ids, diagnostics). Empty selection means: run stock attention.
    """
    poses = np.asarray(poses)
    target = poses[-1]
    now = first_frame - 1  # last generated latent frame
    stored = [i for i in sorted(available) if sink <= i < first_frame - TAIL and i < len(poses)]
    near = [i for i in stored if pose_close(target, poses[i], max_angle_deg, max_distance)]
    info = {"rule": "verified", "stored": len(stored), "pose_matches": len(near), "place_similarity": None,
            "place_verified": None, "trusted_rotation_only": 0, "trusted_by_content": 0, "rejected_untrusted": 0}
    if not near or now < 0:
        return [], info
    current = descriptors.get(now)
    if current is not None:
        here = [j for j in stored if j in descriptors and pose_cost(poses[now], poses[j]) <= verify_cost]
        if here:
            info["place_similarity"] = float(max(current @ descriptors[j] for j in here))
            info["place_verified"] = info["place_similarity"] >= tau_verify
    trusted = []
    for i in near:
        if translation_between(poses, i, now) <= 1e-6:
            info["trusted_rotation_only"] += 1
            trusted.append(i)
        elif info["place_verified"] or (current is not None and i in descriptors
                                        and float(current @ descriptors[i]) >= tau_direct):
            info["trusted_by_content"] += 1
            trusted.append(i)
        else:
            info["rejected_untrusted"] += 1
    ranked = sorted(trusted, key=lambda i: (round(pose_cost(target, poses[i]), 6), i))[:count]
    return sorted(ranked), info


def select_content(descriptors, available, first_frame, sink, count, *, tau=0.75, min_gap=1):
    """Top-`count` stored frames by similarity to the current view, at least `tau`; earliest wins ties."""
    now = first_frame - 1
    current = descriptors.get(now)
    stored = [i for i in sorted(available) if sink <= i < first_frame - TAIL and i in descriptors]
    info = {"rule": "content", "stored": len(stored), "best_similarity": None, "above_threshold": 0}
    if current is None or not stored:
        return [], info
    scored = [(float(current @ descriptors[i]), i) for i in stored]
    info["best_similarity"] = max(score for score, _ in scored)
    good = [(score, i) for score, i in scored if score >= tau]
    info["above_threshold"] = len(good)
    chosen = []
    for score, i in sorted(good, key=lambda item: (-round(item[0], 4), item[1])):
        if all(abs(i - j) >= min_gap for j in chosen):
            chosen.append(i)
        if len(chosen) == count:
            break
    return sorted(chosen), info


class DiskKVStore:
    """One file per latent frame: int16 view of bf16 [layers, 2 (k, v), tokens, heads, head_dim]."""

    def __init__(self, layers, root=None):
        base = Path(root or os.environ.get("WORLDSYNC_KV_DIR") or tempfile.gettempdir())
        self.root = Path(tempfile.mkdtemp(prefix="worldsync-kv-", dir=base))
        self.layers = layers
        self.shape = None
        self.maps = {}
        self.pending = queue.Queue(maxsize=96)
        self.error = None
        self.bytes_written = 0
        self.write_seconds = 0.0
        self.read_seconds = 0.0
        self.frames_read = 0
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _path(self, index):
        return self.root / f"frame-{index:05d}.bin"

    def _drain(self):
        while True:
            item = self.pending.get()
            try:
                if item is None:
                    return
                index, layer, key, value = item
                started = time.monotonic()
                if self.shape is None:
                    self.shape = tuple(key.shape[1:])
                if index not in self.maps:
                    self.maps[index] = np.memmap(self._path(index), dtype=np.int16, mode="w+",
                                                 shape=(self.layers, 2, *self.shape))
                target = self.maps[index]
                target[layer, 0] = key[0].view(torch.int16).numpy()
                target[layer, 1] = value[0].view(torch.int16).numpy()
                if layer == self.layers - 1:
                    target.flush()
                    del self.maps[index]  # later reads reopen read-only
                self.bytes_written += 2 * key[0].numel() * 2
                self.write_seconds += time.monotonic() - started
            except BaseException as error:  # surfaced on the next put/read
                self.error = error
            finally:
                self.pending.task_done()

    def put(self, index, layer, key, value):
        if self.error is not None:
            raise RuntimeError(f"KV archive writer failed: {self.error!r}")
        if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
            raise TypeError("KV archive expects bf16 tensors")
        self.pending.put((index, layer, key, value))

    def read(self, index):
        """Whole frame as a pinned bf16 tensor [layers, 2, tokens, heads, head_dim]."""
        self.pending.join()
        if self.error is not None:
            raise RuntimeError(f"KV archive writer failed: {self.error!r}")
        started = time.monotonic()
        data = np.fromfile(self._path(index), dtype=np.int16).reshape(self.layers, 2, *self.shape)
        tensor = torch.from_numpy(data).view(torch.bfloat16)
        if torch.cuda.is_available():
            tensor = tensor.pin_memory()
        self.read_seconds += time.monotonic() - started
        self.frames_read += 1
        return tensor

    def close(self):
        self.pending.put(None)
        self.thread.join(timeout=120)
        self.maps.clear()
        shutil.rmtree(self.root, ignore_errors=True)


class MemoryV2(BudgetPoseMemory):
    def __init__(self, pipeline, sink_size=6, local_attn_size=18, retrieved=10, rope="native", select="verified",
                 descriptor="dino", stride=1, tau_verify=0.75, tau_direct=0.80, tau_content=0.75,
                 max_angle_deg=60.0, max_distance=2.0, store_root=None, old_archive_frames=24):
        super().__init__(pipeline, sink_size=sink_size, local_attn_size=local_attn_size, retrieved=retrieved,
                         archive_frames=old_archive_frames, rope=rope)
        if select not in ("verified", "content", "pose"):
            raise ValueError("select must be verified, content or pose")
        if descriptor not in ("dino", "thumbnail"):
            raise ValueError("descriptor must be dino or thumbnail")
        if stride < 1:
            raise ValueError("stride must be positive")
        self.select = select
        self.stride = stride
        self.descriptor_name = descriptor
        self.describe = DinoDescriptor() if descriptor == "dino" else None
        self.descriptors = {}
        self.thresholds = {"tau_verify": tau_verify, "tau_direct": tau_direct,
                           "max_angle_deg": max_angle_deg, "max_distance": max_distance}
        self.tau_content = tau_content
        self.store = DiskKVStore(len(pipeline.model.blocks), store_root)
        self.host = {}          # frame id -> pinned bf16 [layers, 2, tokens, heads, head_dim]
        self.old_available = set()  # logical 24-frame archive of the old method (select == "pose")
        self.select_seconds = 0.0
        self.attend_seconds = 0.0
        self.banks = None       # no in-RAM per-layer archive in v2

    def observe(self, chunk, pixels):
        """Record one view descriptor per latent frame of the chunk just decoded."""
        indices = latent_pixel_indices(chunk, len(pixels))
        images = [pixels[i] for i in indices]
        vectors = self.describe(images) if self.describe else [thumbnail_descriptor(image) for image in images]
        for offset, vector in enumerate(vectors):
            self.descriptors[chunk * 4 + offset] = np.asarray(vector, dtype=np.float32)

    def begin(self, chunk, poses):
        started = time.monotonic()
        self.chunk = chunk
        self.poses = np.asarray(poses)
        first_frame = chunk * 4
        self.latest_start = first_frame
        if self.select == "pose":
            self.old_available = evict_redundant(self.poses, self.old_available, self.archive_limit, first_frame - 4)
            self.selected = select_by_pose(self.poses, self.old_available, first_frame, self.sink, self.retrieved)
            info = {"rule": "pose", "old_archive": sorted(self.old_available)}
        elif self.select == "content":
            self.selected, info = select_content(self.descriptors, self.available, first_frame, self.sink,
                                                 self.retrieved, tau=self.tau_content)
        else:
            self.selected, info = select_verified(self.poses, self.descriptors, self.available, first_frame,
                                                  self.sink, self.retrieved, **self.thresholds)
        for index in [i for i in self.host if i not in self.selected]:
            del self.host[index]
        read_before = self.store.read_seconds
        for index in self.selected:
            if index not in self.host:
                self.host[index] = self.store.read(index)
        self.select_seconds += time.monotonic() - started
        self.trace.append({"chunk": chunk, "retrieved_frame_ids": self.selected, "stored_frames": len(self.available),
                           "selection": info, "disk_read_seconds": self.store.read_seconds - read_before,
                           "attention_replaced": False})

    def release_read_cache(self):
        return 0

    def attend(self, query, key, value, **kwargs):
        if self.active is None:
            return self.original_attention(query, key, value, **kwargs)
        layer, start, frame_tokens, grid, freqs = self.active
        self.frame_tokens = frame_tokens
        first_frame = start // frame_tokens
        if self.clean:
            if self.raw_key is None:
                raise RuntimeError("canonical key capture missing")
            raw = self.raw_key.view_as(query)
            v_current = value[:, -query.shape[1]:]
            for offset in range(query.shape[1] // frame_tokens):
                index = first_frame + offset
                if index < self.sink or (index - self.sink) % self.stride:
                    continue
                lo, hi = offset * frame_tokens, (offset + 1) * frame_tokens
                self.store.put(index, layer, raw[:, lo:hi].detach().to("cpu", copy=True),
                               v_current[:, lo:hi].detach().to("cpu", copy=True))
                if layer == 0:
                    self.available.add(index)
                    self.old_available.add(index)
        if key.shape[1] % frame_tokens or key.shape[1] != self.window * frame_tokens or not self.selected:
            return self.original_attention(query, key, value, **kwargs)
        started = time.monotonic()
        canonical = torch.cat([self.host[i][layer, 0] for i in self.selected], dim=0)[None].to(key.device, non_blocking=True)
        retrieved_value = torch.cat([self.host[i][layer, 1] for i in self.selected], dim=0)[None].to(value.device, non_blocking=True)
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
        retrieved_key = retrieved_key.to(key.dtype)
        # Never attend to fewer recent frames than stock would keep after the swap.
        tail = max(TAIL, STOCK_RECENT - len(self.selected))
        combined_key = torch.cat([key[:, :self.sink * frame_tokens], retrieved_key, key[:, -tail * frame_tokens:]], dim=1)
        combined_value = torch.cat([value[:, :self.sink * frame_tokens], retrieved_value, value[:, -tail * frame_tokens:]], dim=1)
        del canonical, retrieved_key, retrieved_value
        self.trace[-1]["attention_replaced"] = True
        self.trace[-1]["tail_latent_frames"] = tail
        self.retrieval_calls += 1
        self.attend_seconds += time.monotonic() - started
        return self.original_attention(query, combined_key, combined_value, **kwargs)

    def metadata(self):
        return {"method": "memory v2 (disk): unlimited exact-KV archive on local disk, content-verified retrieval; diagnostic",
                "kv_source": "disk", "selection_rule": self.select, "descriptor": self.descriptor_name,
                "thresholds": {**self.thresholds, "tau_content": self.tau_content},
                "retrieved_latent_frames": self.retrieved, "retrieved_frame_count_is_limit": True,
                "attended_latent_frames": self.sink + self.retrieved + TAIL, "rope": self.rope,
                "archive_stride": self.stride, "stored_latent_frames": len(self.available),
                "disk_bytes_written": self.store.bytes_written, "disk_write_seconds": self.store.write_seconds,
                "disk_read_seconds": self.store.read_seconds, "disk_frames_read": self.store.frames_read,
                "selection_and_read_seconds": self.select_seconds, "retrieval_attend_overhead_seconds": self.attend_seconds,
                "host_bytes_retrieved_now": sum(t.numel() * 2 for t in self.host.values()),
                "pose_source": "commanded poses, verified by content unless only rotation was commanded since",
                "retrieval_attention_calls": self.retrieval_calls, "trace": self.trace}

    def close(self):
        try:
            super().close()
        finally:
            self.host.clear()
            self.store.close()
