"""Training-free full-channel group quantization for isolated LingBot KV tests.

This reuses only the validated segmented pre-RoPE lifecycle from NativeNanoKV.
No fitted projection, reader, backbone weights, or inference defaults change.
Stored scales are BF16; rounding and reconstruction arithmetic use FP32.
"""

import torch

from worldsync.lingbot_nanokv import NativeNanoKV

FORMAT = "lingbot_full_channel_group_quant_v1"
_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def _settings(bits, group_size):
    if type(bits) is not int or bits not in (4, 8):
        raise ValueError("Group quantization supports INT4 or INT8")
    if type(group_size) is not int or group_size not in (2, 4, 8, 16, 32, 64, 128):
        raise ValueError("Group size must be a power of two in [2,128]")


@torch.no_grad()
def encode_group_tensor(value, *, bits=4, group_size=32):
    """Quantize independent groups within each token/head; keep every channel."""
    _settings(bits, group_size)
    if (
        not torch.is_tensor(value)
        or value.ndim != 4
        or value.dtype not in _DTYPES
        or any(d <= 0 for d in value.shape)
        or value.shape[-1] % group_size
    ):
        raise ValueError("Expected floating [B,S,H,D] with complete per-head groups")
    maximum = (1 << (bits - 1)) - 1
    grouped = value.float().reshape(
        *value.shape[:-1], value.shape[-1] // group_size, group_size
    )
    amax = grouped.abs().amax(-1)
    if not bool(torch.isfinite(amax).all()):
        raise ValueError("Cannot quantize nonfinite KV")
    # A normal BF16 floor avoids zero scales for zeros and subnormal groups.
    scale = (
        (amax / maximum).clamp_min(torch.finfo(torch.bfloat16).tiny).to(torch.bfloat16)
    )
    codes = (grouped / scale.float().unsqueeze(-1)).round().clamp(-maximum, maximum)
    codes = codes.to(torch.int8).reshape(value.shape)
    if bits == 4:
        unsigned = (codes + 8).to(torch.uint8)
        codes = unsigned[..., 0::2] | (unsigned[..., 1::2] << 4)
    return {
        "format": FORMAT,
        "shape": tuple(value.shape),
        "bits": bits,
        "group_size": group_size,
        "codes": codes,
        "scales": scale,
    }


def _validate_parts(parts):
    if not isinstance(parts, (list, tuple)) or not parts:
        raise ValueError("At least one quantized segment is required")
    if any(not isinstance(part, dict) for part in parts):
        raise ValueError("Quantized segments must be dictionaries")
    first = parts[0]
    bits, group_size = first.get("bits"), first.get("group_size")
    _settings(bits, group_size)
    shape = first.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        raise ValueError("Malformed quantized segment shape")
    batch, _, heads, dim = shape
    if any(type(d) is not int or d <= 0 for d in shape) or dim % group_size:
        raise ValueError("Malformed quantized segment dimensions")
    for part in parts:
        current = part.get("shape")
        if (
            part.get("format") != FORMAT
            or part.get("bits") != bits
            or part.get("group_size") != group_size
            or not isinstance(current, (list, tuple))
            or len(current) != 4
            or any(type(d) is not int or d <= 0 for d in current)
            or (current[0], current[2], current[3]) != (batch, heads, dim)
        ):
            raise ValueError("Incompatible quantized segments")
        codes, scales = part.get("codes"), part.get("scales")
        expected = (batch, current[1], heads, dim // (2 if bits == 4 else 1))
        if (
            not torch.is_tensor(codes)
            or tuple(codes.shape) != expected
            or codes.dtype != (torch.uint8 if bits == 4 else torch.int8)
            or not torch.is_tensor(scales)
            or tuple(scales.shape) != (batch, current[1], heads, dim // group_size)
            or scales.dtype != torch.bfloat16
        ):
            raise ValueError("Malformed quantized codes/scales")
    return bits, group_size, batch, heads, dim


@torch.no_grad()
def decode_group_tensors(parts, *, dtype, device):
    """Concatenate compact segments, then decode once per K or V history."""
    if dtype not in _DTYPES:
        raise ValueError("Unsupported reconstruction dtype")
    bits, group_size, batch, heads, dim = _validate_parts(parts)
    codes = torch.cat([part["codes"].to(device) for part in parts], dim=1)
    scales = torch.cat([part["scales"].to(device) for part in parts], dim=1).float()
    if not bool((torch.isfinite(scales) & (scales > 0)).all()):
        raise ValueError("Quantized scales must be finite and positive")
    if bits == 4:
        low, high = (codes & 15).to(torch.int8) - 8, (codes >> 4).to(torch.int8) - 8
        codes = torch.stack((low, high), dim=-1).reshape(batch, -1, heads, dim)
    groups = codes.float().reshape(batch, -1, heads, dim // group_size, group_size)
    reconstructed = (groups * scales.unsqueeze(-1)).reshape(batch, -1, heads, dim)
    # BF16 scale rounding at dtype maxima can slightly overshoot. Saturating
    # reconstruction prevents infinities without changing the stored codes.
    limit = torch.finfo(dtype).max
    return reconstructed.clamp(-limit, limit).to(dtype)


class GroupQuantKV(NativeNanoKV):
    """Native D128 inference only; CPU toys may use fewer heads/layers.

    The public 14B model has 40 heads and 40 layers; quantization acts on each
    head independently, so smaller native CPU models test the same mechanism.
    """

    def __init__(
        self, pipeline, source_root, *, capacity_frames=58, bits=4, group_size=32
    ):
        _settings(bits, group_size)
        self.bits, self.group_size = bits, group_size
        super().__init__(
            pipeline,
            source_root=source_root,
            capacity_frames=capacity_frames,
            identity=True,
        )

    def _helpers(self):
        super()._helpers()
        self.runtime.encode_pair = self._encode_pair
        self.runtime.decode_segments = self._decode_segments

    def _encode_pair(self, key, value, *, cache):
        if key.shape != value.shape:
            raise ValueError("Quantized K/V must share native dimensions")
        return {
            "global_start": cache["codec_commit_start"],
            "global_end": cache["codec_commit_start"] + key.shape[1],
            "k": encode_group_tensor(key, bits=self.bits, group_size=self.group_size),
            "v": encode_group_tensor(value, bits=self.bits, group_size=self.group_size),
        }

    @staticmethod
    def _decode_segments(segments, *, dtype, device):
        return tuple(
            decode_group_tensors(
                [segment[target] for segment in segments], dtype=dtype, device=device
            )
            for target in ("k", "v")
        )

    def __enter__(self):
        if not self.model.blocks or any(
            block.self_attn.head_dim != 128 for block in self.model.blocks
        ):
            raise ValueError(
                "GroupQuantKV native inference requires 128 channels per head"
            )
        return super().__enter__()

    def metadata(self):
        result = super().metadata()
        result.update(
            {
                "method": "full-channel symmetric group quantization",
                "identity": False,
                "codec_sha256": None,
                "group_bits": self.bits,
                "group_size": self.group_size,
                "scale_dtype": "bfloat16",
                "all_head_channels_retained": True,
                "fitted_operators": False,
                "shape": {
                    "layers": len(self.model.blocks),
                    "heads": self.model.blocks[0].self_attn.num_heads,
                    "head_dim": self.model.blocks[0].self_attn.head_dim,
                },
                "quantization_source": "worldsync.lingbot_group_quant; pinned NanoKV lifecycle only",
            }
        )
        return result
