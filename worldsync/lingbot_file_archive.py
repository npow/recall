"""Isolated lossless CPU feature archive backed by flushed, evictable files."""
import mmap
from pathlib import Path
import time

import numpy as np
import torch

from worldsync.lingbot_retrieval import PoseKVMemory


class FileArchiveBank(dict):
    def __init__(self, path, first_frame=6, stop_frame=22):
        super().__init__()
        self.path = Path(path)
        self.first_frame, self.stop_frame = first_frame, stop_frame
        self.mapping = None
        self.dirty = False
        self.bytes_written = 0
        self.flush_wall_s = 0.
        self.evictions = 0
        self.lookup_calls = 0

    @torch.no_grad()
    def __setitem__(self, index, pair):
        if not self.first_frame <= index < self.stop_frame:
            raise ValueError("File archive index exceeds its bounded capacity")
        first, second = pair
        if (first.device.type != "cpu" or second.device.type != "cpu"
                or first.shape != second.shape or first.dtype != second.dtype):
            raise ValueError("File archive requires matching CPU key/value tensors")
        frame_bytes = first.numel() * first.element_size()
        if self.mapping is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.shape, self.dtype = first.shape, first.dtype
            slots = (self.stop_frame - self.first_frame) * 2
            self.mapping = np.memmap(self.path, dtype=np.uint8, mode="w+", shape=(slots, frame_bytes))
            if not hasattr(self.mapping._mmap, "madvise"):
                raise RuntimeError("Local file archive requires Linux mmap page eviction")
        if first.shape != self.shape or first.dtype != self.dtype:
            raise ValueError("File archive shape changed within one episode")
        slots = []
        for which, tensor in enumerate(pair):
            position = (index - self.first_frame) * 2 + which
            view = torch.from_numpy(np.asarray(self.mapping[position])).view(self.dtype).reshape(self.shape)
            view.copy_(tensor)
            slots.append(view)
            self.bytes_written += frame_bytes
        super().__setitem__(index, tuple(slots))
        self.dirty = True

    def __getitem__(self, index):
        self.lookup_calls += 1
        return super().__getitem__(index)

    def flush_and_evict(self):
        if self.mapping is None:
            return
        if self.dirty:
            started = time.monotonic()
            self.mapping.flush()
            self.flush_wall_s += time.monotonic() - started
            self.dirty = False
        self.mapping._mmap.madvise(mmap.MADV_DONTNEED)
        self.evictions += 1

    def metadata(self):
        return {"file": str(self.path), "mapped_bytes": self.mapping.nbytes if self.mapping is not None else 0,
                "logical_bytes_written": self.bytes_written, "flush_wall_s": self.flush_wall_s,
                "page_evictions": self.evictions, "pair_lookups": self.lookup_calls}

    def close(self):
        self.flush_and_evict()
        # Tensor views cannot be used after close; drop owned views before unmap.
        self.clear()
        if self.mapping is not None:
            self.mapping._mmap.close()
            self.mapping = None


class FileArchiveMemory(PoseKVMemory):
    def __init__(self, pipeline, *args, archive_root, **kwargs):
        self.archive_root = Path(archive_root)
        self.final_metadata = None
        super().__init__(pipeline, *args, **kwargs)
        if (self.cache_frames != 18 or self.retrieved_frames != 4 or self.selection != "nearest"
                or self.key_positions != "virtual" or self.history_weight is not None):
            self.close()
            raise ValueError("Local file archive validates only frozen18-frame/four-nearest virtual retrieval")

    def attend(self, query, key, value, **kwargs):
        bank = None
        if self.active is not None:
            layer = self.active[0]
            if layer not in self.banks:
                self.banks[layer] = FileArchiveBank(self.archive_root / f"layer-{layer:03d}.bin", *self.archive_range)
            bank = self.banks[layer]
        try:
            return super().attend(query, key, value, **kwargs)
        finally:
            if bank is not None:
                bank.flush_and_evict()

    def metadata(self):
        result = super().metadata()
        result["archive_storage"] = {"kind": "lossless file-backed original dtype",
            "private_archive_tensor_payload_bytes": 0,
            "residency_policy": "flush dirty pages and MADV_DONTNEED after each attention layer call",
            "caveat": "File-backed pages and filesystem caches can still consume reclaimable physical RAM",
            "banks": [bank.metadata() for bank in self.banks.values()]}
        return result

    def close(self):
        self.final_metadata = self.metadata()
        super().close()
        for bank in self.banks.values():
            bank.close()
        self.banks.clear()
