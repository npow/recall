"""Isolated local-memory prototypes; never installed by the normal rollout.

Quantization changes historical features and needs separate quality evaluation.
The one-shot encoder wrapper deliberately cannot support a second generation.
"""
from dataclasses import dataclass

import torch

from worldsync.lingbot_retrieval import PoseKVMemory


@dataclass
class PackedArchiveTensor:
    codes: torch.Tensor
    scales: torch.Tensor

    @classmethod
    def pack(cls, tensor):
        if tensor.device.type != "cpu" or not torch.is_floating_point(tensor):
            raise ValueError("Archive quantization expects a floating CPU tensor")
        values = tensor.detach().float()
        if not torch.isfinite(values).all():
            raise ValueError("Cannot quantize non-finite archive values")
        maximum = values.abs().amax(dim=-1, keepdim=True)
        scales = torch.where(maximum == 0, torch.ones_like(maximum),
                             (maximum / 127).clamp_min(torch.finfo(torch.float32).tiny))
        codes = torch.round(values / scales).clamp(-127, 127).to(torch.int8)
        return cls(codes, scales)

    def dequantize(self, device, dtype):
        # Each archived frame is transferred/dequantized independently. Avoid a
        # full-archive FP32 temporary; the read output is the original dtype.
        return (self.codes.to(device=device).float()
                * self.scales.to(device=device)).to(dtype=dtype)

    def numel(self):
        return self.codes.numel()

    def element_size(self):
        return self.codes.element_size()

    @property
    def scale_bytes(self):
        return self.scales.numel() * self.scales.element_size()

    @property
    def storage_bytes(self):
        return self.numel() * self.element_size() + self.scale_bytes


class QuantizedArchiveBank(dict):
    def __setitem__(self, index, pair):
        super().__setitem__(index, tuple(PackedArchiveTensor.pack(tensor) for tensor in pair))


class Int8ArchiveMemory(PoseKVMemory):
    """Prototype: same selection/native KV, int8 CPU history, BF16 GPU reads."""
    def __init__(self, pipeline, key_positions="virtual", retrieved_frames=4,
                 selection="nearest", cache_frames=18, history_weight=None):
        if (key_positions != "virtual" or retrieved_frames != 4 or selection != "nearest"
                or cache_frames != 18 or history_weight is not None):
            raise ValueError("Local quantization prototype is limited to the frozen18-frame/four-nearest policy")
        super().__init__(pipeline, key_positions, retrieved_frames, selection, cache_frames, history_weight)

    def attend(self, query, key, value, **kwargs):
        if self.active is not None:
            layer, start, frame_tokens, grid, freqs = self.active
            bank = self.banks.setdefault(layer, QuantizedArchiveBank())
            if (len(self.selected) >= 4 and key.shape[1] == self.cache_frames * frame_tokens
                    and layer not in self.cached):
                if any(index not in bank for index in self.selected):
                    raise RuntimeError("Quantized history unavailable in an attention layer")
                canonical = torch.cat([bank[index][0].dequantize(key.device, key.dtype)
                                       for index in self.selected], dim=1)
                retrieved_value = torch.cat([bank[index][1].dequantize(value.device, value.dtype)
                                             for index in self.selected], dim=1)
                virtual_grid = grid.clone()
                virtual_grid[:, 0] = len(self.selected)
                with torch.amp.autocast("cuda", enabled=False):
                    retrieved_key = self.module.causal_rope_apply(canonical, virtual_grid, freqs,
                        start_frame=start // frame_tokens + 10 - self.cache_frames).to(key.dtype)
                self.cached[layer] = retrieved_key, retrieved_value
        # Parent archives current clean features into our compressed mapping and
        # consumes the prepopulated GPU read cache. All attention composition,
        # native cache writes, selection and cleanup remain in the frozen method.
        return super().attend(query, key, value, **kwargs)

    def metadata(self):
        result = super().metadata()
        scales = sum(tensor.scale_bytes for bank in self.banks.values() for pair in bank.values() for tensor in pair)
        result["archive_bytes"] += scales
        result["archive_quantization"] = {"codes": "symmetric int8", "scales": "FP32 per token/head vector",
                                           "scale_bytes": scales, "dequantization": "on GPU at read, original BF16 dtype",
                                           "quality_validated": False}
        return result


class OneShotTextEncoder:
    """Discard the completed T5 payload before VAE/KV allocation; no reload path."""
    def __init__(self, encoder):
        self.encoder = encoder
        self.model = encoder.model
        self.calls = 0
        self.released_parameter_bytes = 0

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        if self.calls:
            raise RuntimeError("One-shot text encoder payload was discarded; reload before another request")
        result = self.encoder(*args, **kwargs)
        self.calls += 1
        self.released_parameter_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        self.model.to("meta")
        # The frozen rollout calls model.cpu() at its first denoiser hook. A
        # meta skeleton has nothing to offload; retain that call's interface.
        self.model.cpu = lambda: self.model
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        return result

    def __getattr__(self, name):
        return getattr(self.encoder, name)


def archive_storage_bytes(bf16_bytes, head_dimension=128):
    elements = bf16_bytes // 2
    if elements % head_dimension:
        raise ValueError("Archive does not contain complete head vectors")
    return elements + elements // head_dimension * 4


def memory_budget(available_bytes, gpu_peak_bytes, archive_bf16_bytes,
                  process_overhead_bytes=5_000_000_000, allocator_headroom_bytes=6_000_000_000,
                  reserve_bytes=16 * 1024 ** 3):
    archive = archive_storage_bytes(archive_bf16_bytes)
    estimate = gpu_peak_bytes + archive + process_overhead_bytes + allocator_headroom_bytes
    return {"available_bytes": available_bytes, "estimated_app_bytes": estimate,
            "compressed_archive_bytes": archive, "uncompressed_archive_bytes": archive_bf16_bytes,
            "required_system_reserve_bytes": reserve_bytes,
            "estimated_remaining_bytes": available_bytes - estimate,
            "conservative_budget_fits": available_bytes - estimate >= reserve_bytes,
            "caveat": "Discrete-GPU peak is a conservative proxy, not a measured local unified-memory peak; no launch permission implied."}
