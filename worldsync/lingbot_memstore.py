"""Cheaper storage for archived memory frames (ablation of bounded pose memory; diagnostic only).

Exact storage keeps canonical bf16 K and V per layer: 2 tensors x 1560 tokens x 5120 dims x 2 B x 40 layers,
about 1.28 GB per latent frame. Each codec below stores less and reconstructs K/V right before attention:

* fp8 / int8      8-bit values (+ a scale): about half of bf16 (bf16 -> 8 bit is 2x, not 4x);
* pool2 / pool4   spatial average over the 30x52 token grid, upsampled (repeated) at read, so the attended
                  token count and the RoPE positions are unchanged: 4x / about 15x smaller;
* top25 / top10   keep the tokens with the largest value norm (K is RMS-normalised, so its norm does not rank
                  tokens); rand25 is the control; attended memory tokens shrink accordingly;
* lr64 / lr256    randomised rank-r factorisation of K and V per layer and frame;
* `a+b`           composition, e.g. `pool2+fp8`;
* layer subsets   are handled by the memory class, not the codec (storage scales with the fraction kept).

Reads keep the stored (compressed) form on the GPU when it is small, otherwise on pinned CPU memory, and decode
per attention call, so the GPU never holds a decoded bf16 copy of all retrieved frames for all layers.
"""
import math

import torch

FP8_MAX = 448.0
EXACT_BYTES_PER_LAYER = 2 * 1560 * 5120 * 2  # K and V, bf16, one latent frame at 480x832

LAYER_SETS = {
    "all": lambda n: list(range(n)),
    "even": lambda n: list(range(0, n, 2)),
    "every4": lambda n: list(range(0, n, 4)),
    "first10": lambda n: list(range(0, n // 4)),
    "last10": lambda n: list(range(n - n // 4, n)),
    "first20": lambda n: list(range(0, n // 2)),
    "last20": lambda n: list(range(n // 2, n)),
    "mid20": lambda n: list(range(n // 4, n - n // 4)),
}


def layer_set(name, layers):
    if name not in LAYER_SETS:
        raise ValueError(f"unknown layer subset {name}")
    return LAYER_SETS[name](layers)


def nbytes(stored):
    return sum(t.numel() * t.element_size() for t in stored.values() if torch.is_tensor(t))


class Exact:
    name = "exact"

    def encode(self, k, v, hw):
        return {"k": k.detach().to("cpu", copy=True), "v": v.detach().to("cpu", copy=True)}

    def decode(self, s, dtype):
        return s["k"].to(dtype), s["v"].to(dtype), None


class Fp8:
    """float8_e4m3fn values with one scale per token and head."""
    name = "fp8"

    @staticmethod
    def _q(x):
        amax = x.detach().abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-8)
        scale = amax / FP8_MAX
        return (x.float() / scale).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), scale.to(torch.bfloat16)

    def encode(self, k, v, hw):
        kq, ks = self._q(k)
        vq, vs = self._q(v)
        return {"k": kq.cpu(), "ks": ks.cpu(), "v": vq.cpu(), "vs": vs.cpu()}

    def decode(self, s, dtype):
        return s["k"].to(dtype) * s["ks"].to(dtype), s["v"].to(dtype) * s["vs"].to(dtype), None


class Int8:
    """Symmetric int8 with one scale per token (shared by all heads and dims)."""
    name = "int8"

    @staticmethod
    def _q(x):
        scale = x.detach().abs().flatten(2).amax(dim=-1).float().clamp_min(1e-8)[..., None, None] / 127.0
        return torch.round(x.float() / scale).clamp_(-127, 127).to(torch.int8), scale

    def encode(self, k, v, hw):
        kq, ks = self._q(k)
        vq, vs = self._q(v)
        return {"k": kq.cpu(), "ks": ks.cpu(), "v": vq.cpu(), "vs": vs.cpu()}

    def decode(self, s, dtype):
        return (s["k"].float() * s["ks"]).to(dtype), (s["v"].float() * s["vs"]).to(dtype), None


class Pool:
    """Average over p x p token blocks; repeat the block mean at read (same token count and positions)."""

    def __init__(self, p, inner=None):
        self.p, self.inner = p, inner or Exact()
        self.name = f"pool{p}" + ("" if inner is None else f"+{inner.name}")

    def _pool(self, x, hw):
        h, w = hw
        b, t, n, d = x.shape
        grid = x.float().reshape(b, h, w, n * d).permute(0, 3, 1, 2)
        pooled = torch.nn.functional.avg_pool2d(grid, self.p, ceil_mode=True, count_include_pad=False)
        return pooled.permute(0, 2, 3, 1).reshape(b, -1, n, d).to(x.dtype)

    def encode(self, k, v, hw):
        stored = self.inner.encode(self._pool(k, hw), self._pool(v, hw), None)
        stored["hw"] = hw
        return stored

    def _up(self, x, hw):
        h, w = hw
        ph, pw = math.ceil(h / self.p), math.ceil(w / self.p)
        b, t, n, d = x.shape
        grid = x.reshape(b, ph, pw, n, d).repeat_interleave(self.p, dim=1).repeat_interleave(self.p, dim=2)
        return grid[:, :h, :w].reshape(b, h * w, n, d)

    def decode(self, s, dtype):
        k, v, _ = self.inner.decode(s, dtype)
        return self._up(k, s["hw"]), self._up(v, s["hw"]), None


class TopTokens:
    """Keep a fraction of tokens: largest value norm (`vnorm`) or a fixed pseudo-random subset (`random`)."""

    def __init__(self, fraction, by="vnorm"):
        if by not in ("vnorm", "random"):
            raise ValueError("by must be vnorm or random")
        self.fraction, self.by = fraction, by
        self.name = f"{'top' if by == 'vnorm' else 'rand'}{round(fraction * 100)}"

    def encode(self, k, v, hw):
        tokens = k.shape[1]
        keep = max(1, round(tokens * self.fraction))
        if self.by == "vnorm":
            index = v.detach().float().flatten(2).norm(dim=-1)[0].topk(keep).indices.sort().values
        else:
            generator = torch.Generator().manual_seed(tokens)
            index = torch.randperm(tokens, generator=generator)[:keep].sort().values.to(k.device)
        return {"k": k[:, index].detach().cpu(), "v": v[:, index].detach().cpu(), "index": index.to(torch.int32).cpu(),
                "tokens": tokens}

    def decode(self, s, dtype):
        return s["k"].to(dtype), s["v"].to(dtype), s["index"].long()


class LowRank:
    """Randomised rank-r factorisation of the [tokens, heads*dim] matrix, for K and for V."""

    def __init__(self, rank):
        self.rank = rank
        self.name = f"lr{rank}"

    def _f(self, x):
        b, t, n, d = x.shape
        # The clean pass runs under bf16 autocast, where the QR inside svd_lowrank is not implemented.
        with torch.amp.autocast(x.device.type, enabled=False):
            u, s, v = torch.svd_lowrank(x[0].detach().to(torch.float32).reshape(t, n * d), q=self.rank, niter=2)
            return (u * s).to(torch.float16).cpu(), v.T.contiguous().to(torch.float16).cpu()

    def encode(self, k, v, hw):
        kl, kr = self._f(k)
        vl, vr = self._f(v)
        return {"kl": kl, "kr": kr, "vl": vl, "vr": vr, "heads": k.shape[2]}

    def decode(self, s, dtype):
        def build(left, right):
            full = left.float() @ right.float()
            return full.view(1, full.shape[0], s["heads"], -1).to(dtype)
        return build(s["kl"], s["kr"]), build(s["vl"], s["vr"]), None


def make_codec(name):
    """`exact`, `fp8`, `int8`, `pool2`, `pool4`, `top25`, `top10`, `rand25`, `lr64`, `lr256`, or `poolN+fp8|int8`."""
    parts = name.split("+")
    simple = {"exact": Exact, "fp8": Fp8, "int8": Int8}
    if len(parts) == 2:
        if not parts[0].startswith("pool") or parts[1] not in ("fp8", "int8"):
            raise ValueError("only poolN+fp8 / poolN+int8 compositions are supported")
        return Pool(int(parts[0][4:]), simple[parts[1]]())
    if len(parts) != 1:
        raise ValueError(f"unknown storage codec {name}")
    if name in simple:
        return simple[name]()
    if name.startswith("pool"):
        return Pool(int(name[4:]))
    if name.startswith("top"):
        return TopTokens(int(name[3:]) / 100)
    if name.startswith("rand"):
        return TopTokens(int(name[4:]) / 100, by="random")
    if name.startswith("lr"):
        return LowRank(int(name[2:]))
    raise ValueError(f"unknown storage codec {name}")


def to_device(stored, device, pin=False):
    out = {}
    for name, value in stored.items():
        if not torch.is_tensor(value):
            out[name] = value
        elif device == "cpu":
            out[name] = value.pin_memory() if pin else value
        else:
            out[name] = value.to(device, non_blocking=True)
    return out


def positioned_keys(rope, canonical, index, tokens, position, grid, freqs):
    """Apply RoPE for one latent frame at temporal `position`; `index` selects a token subset (or None)."""
    single = grid.clone()
    single[:, 0] = 1
    if index is None:
        return rope(canonical, single, freqs, start_frame=position)
    full = canonical.new_zeros((canonical.shape[0], tokens) + tuple(canonical.shape[2:]))
    full[:, index] = canonical
    return rope(full, single, freqs, start_frame=position)[:, index]
