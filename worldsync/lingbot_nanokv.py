"""Opt-in, instance-scoped NanoKV compression for the pinned native LingBot v2.

The released codec is lossy. Identity mode proves the cache lifecycle separately;
neither mode changes the generator's weights, camera, sampler, or defaults.
"""

import hashlib
import importlib.util
import inspect
import math
import sys
import types
import uuid
from pathlib import Path

import torch

CODEC_SHA256 = "b014909923f657b8019203bd581da77e8c6e0880bf7aa80b3eadfbf6a433f458"
HELPER_HASHES = {
    "lingbot_v2_cache_runtime.py": "db3a7f1d6535b85399fdfe9d65827ab7efa156d589a2430f71d65672fe6fd075",
    "lingbot_v2_kv_codec.py": "a3fa8a174531879089ae8f9898182cecaf84368d5a932758169f46a1bc66f109",
    "lingbot_v2_kv_codec_triton.py": "8729c83c71e8b60545e13509a22e8b68d129c65bb1b8c57e7b3e3d45eba9efb4",
}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(map(_tensor_bytes, value))
    return 0


def validate_codec(path):
    """Verify the published file before deserializing tensors on CPU only."""
    if _sha(path) != CODEC_SHA256:
        raise ValueError("NanoKV codec SHA256 mismatch")
    packed = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    expected = {
        f"rank00:{target}:{layer:02d}" for target in ("k", "v") for layer in range(40)
    }
    if (
        packed.get("format") != "lingbot_v2_operator_int8_v1"
        or packed.get("rank") != 640
        or packed.get("layers") != 40
        or packed.get("sp_ranks") != 1
        or packed.get("k_a1_scale") != 1.0
        or set(packed.get("entries", {})) != expected
    ):
        raise ValueError("NanoKV packed codec metadata mismatch")
    for entry in packed["entries"].values():
        tensors = [(entry["mean"], (5120,), torch.float32)]
        for name, shape in (("encoder", (640, 5120)), ("decoder", (5120, 640))):
            tensors.extend(
                (
                    (entry[name]["qweight"], shape, torch.int8),
                    (entry[name]["scale"], (shape[0], 1), torch.bfloat16),
                )
            )
        if any(
            tuple(t.shape) != shape or t.dtype != dtype or t.device.type != "cpu"
            for t, shape, dtype in tensors
        ):
            raise ValueError("NanoKV packed codec tensor shape/dtype mismatch")
        if any(
            not torch.isfinite(t).all()
            for t, _, dtype in tensors
            if dtype != torch.int8
        ):
            raise ValueError("NanoKV packed codec has nonfinite coefficients")
    return packed


class NativeNanoKV:
    def __init__(
        self,
        pipeline,
        *,
        source_root,
        codec_path=None,
        capacity_frames=58,
        identity=False,
    ):
        if (
            type(capacity_frames) is not int
            or capacity_frames < 10
            or (capacity_frames - 6) % 4
        ):
            raise ValueError(
                "NanoKV capacity must preserve six sinks plus complete four-frame blocks"
            )
        self.pipeline, self.model = pipeline, pipeline.model
        self.source_root = Path(source_root)
        self.codec_path = None if codec_path is None else Path(codec_path)
        self.capacity_frames, self.identity = capacity_frames, identity
        self.bindings, self.handles, self.caches, self.trace = [], [], [], []
        self.operator_cache = {}
        self.clean = False
        self.closed = self.active = False
        self.namespace = "_worldsync_nanokv_" + uuid.uuid4().hex
        self.source_hashes = {}
        self.packed = None
        self.payload_peak_bytes = self.operator_bytes = 0
        self.allocator_binding = None

    def _helpers(self):
        root = self.source_root / "models/LingBot-World-2.0/compression/wan/modules"
        for filename, expected in HELPER_HASHES.items():
            actual = _sha(root / filename)
            if actual != expected:
                raise ValueError(f"Pinned NanoKV helper changed: {filename}")
            self.source_hashes[filename] = actual
        package = types.ModuleType(self.namespace)
        package.__path__ = [str(root.resolve())]
        sys.modules[self.namespace] = package
        for stem in ("lingbot_v2_kv_codec", "lingbot_v2_cache_runtime"):
            name = self.namespace + "." + stem
            spec = importlib.util.spec_from_file_location(name, root / (stem + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            setattr(self, "codec" if stem.endswith("codec") else "runtime", module)
        # Replace artifact I/O only, preserving the released codec arithmetic.
        # No environment variables or global nanokv/wan namespaces are changed.
        self.codec._load_operator = self._operator
        if self.identity:
            self.runtime.encode_pair = self._identity_encode
            self.runtime.decode_segments = self._identity_decode

    def _operator(
        self, root, *, target, sp_rank, layer, joint_dim, rank, device, linear_root=None
    ):
        if (
            sp_rank != 0
            or joint_dim != 5120
            or rank != 640
            or target not in ("k", "v")
            or not 0 <= layer < 40
        ):
            raise ValueError(
                "NanoKV operator request differs from the verified checkpoint"
            )
        key = (target, layer, str(device))
        if key not in self.operator_cache:
            entry = self.packed["entries"][f"rank00:{target}:{layer:02d}"]
            result = {"mean": entry["mean"].to(device=device, dtype=torch.float32)}
            for name in ("encoder", "decoder"):
                result[name] = {
                    "qweight": entry[name]["qweight"].to(device),
                    "scale": entry[name]["scale"].to(
                        device=device, dtype=torch.bfloat16
                    ),
                }
            self.operator_cache[key] = result
            self.operator_bytes = _tensor_bytes(self.operator_cache)
        return self.operator_cache[key]

    @staticmethod
    def _identity_encode(key, value, *, cache):
        return {
            "global_start": cache["codec_commit_start"],
            "global_end": cache["codec_commit_start"] + key.shape[1],
            "k": key.detach().clone(),
            "v": value.detach().clone(),
        }

    @staticmethod
    def _identity_decode(segments, *, dtype, device):
        return tuple(
            torch.cat([segment[target] for segment in segments], dim=1).to(
                device=device, dtype=dtype
            )
            for target in ("k", "v")
        )

    def _allocate(self, original, num_layers, shape, dtype, device):
        if self.caches:
            raise ValueError("Use a fresh NanoKV context for each generated episode")
        if (
            num_layers != len(self.bindings)
            or shape[0] != 1
            or tuple(shape[2:])
            != (self.bindings[0][0].num_heads, self.bindings[0][0].head_dim)
        ):
            raise ValueError(
                "NanoKV requires native single-rank batch-one cache shapes"
            )
        placeholder = list(shape)
        placeholder[1] = 1
        caches = original(
            num_layers=num_layers, shape=placeholder, dtype=dtype, device=device
        )
        for layer, cache in enumerate(caches):
            self.runtime.configure_cache(
                cache,
                layer=layer,
                sp_rank=0,
                options={
                    "codec_enabled": True,
                    "codec_capacity_tokens": int(shape[1]),
                    "codec_rank": 640,
                    "codec_clusters": 64,
                    "codec_iters": 2,
                    "codec_basis_root_k": "verified-packed-codec",
                    "codec_basis_root_v": "verified-packed-codec",
                },
            )
            self.caches.append(cache)
        return caches

    def _forward(
        self,
        layer,
        original,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        kv_cache=None,
        current_start=0,
        max_attention_size=1_000_000,
        frame_seqlen=None,
        seq_lens_int=None,
    ):
        if not self.active:
            raise RuntimeError("NanoKV attention used outside its active context")
        if torch.is_grad_enabled() or self.model.training:
            raise ValueError("NanoKV diagnostic supports frozen eval inference only")
        b, s = x.shape[:2]
        if b != 1 or not any(kv_cache is cache for cache in self.caches):
            raise ValueError("NanoKV attention needs this adapter's allocated cache")
        frame_seqlen = (
            math.prod(grid_sizes[0][1:].tolist())
            if frame_seqlen is None
            else frame_seqlen
        )
        capacity = self.capacity_frames * frame_seqlen
        if (
            kv_cache["codec_capacity_tokens"] != capacity
            or max_attention_size != capacity
            or s != 4 * frame_seqlen
        ):
            raise ValueError(
                "NanoKV logical cache/chunk shape differs from its frozen configuration"
            )
        shape = (b, s, layer.num_heads, layer.head_dim)
        query = layer.norm_q(layer.q(x)).view(shape)
        key = layer.norm_k(layer.k(x)).view(shape)
        value = layer.v(x).view(shape)
        globals_ = original.__func__.__globals__
        query = globals_["causal_rope_apply"](
            query, grid_sizes, freqs, start_frame=current_start // frame_seqlen
        ).type_as(value)
        kv_cache["codec_commit"] = self.clean
        keys, values, positions = self.runtime.codec_cache_attention_inputs(
            kv_cache,
            key,
            value,
            current_start=int(current_start),
            capacity=capacity,
            sink_tokens=6 * frame_seqlen,
            max_attention_size=capacity,
        )
        keys = self.runtime.rope_pre_key_by_positions(
            keys,
            positions,
            grid_sizes=grid_sizes,
            freqs=freqs,
            frame_seqlen=frame_seqlen,
        )
        self.trace.append(
            {
                "layer": kv_cache["codec_layer"],
                "clean": self.clean,
                "current_start": int(current_start),
                "query_tokens": s,
                "read_key_tokens": keys.shape[1],
                "capacity_tokens": capacity,
                "committed_end": kv_cache["codec_committed_end"],
            }
        )
        self.payload_peak_bytes = max(
            self.payload_peak_bytes,
            sum(_tensor_bytes(cache["codec_segments"]) for cache in self.caches),
        )
        return layer.o(globals_["attention"](query, keys, values).flatten(2))

    def _before(self, module, positional, keywords):
        if any(
            hasattr(layer, "memory_reader")
            or native.__func__.__globals__["attention"] is not self.attention
            for layer, native in self.bindings
        ):
            raise ValueError(
                "Attention/reader changed during isolated NanoKV inference"
            )
        timestep = keywords.get("t", positional[1] if len(positional) > 1 else None)
        if timestep is None:
            raise ValueError("NanoKV requires explicit native clean-pass timestep")
        self.clean = bool(torch.all(timestep == 0))

    def _after(self, *_):
        self.clean = False
        for cache in self.caches:
            cache["codec_commit"] = False

    def __enter__(self):
        if self.closed or self.active:
            raise RuntimeError("NanoKV contexts cannot be reused")
        try:
            if getattr(self.pipeline, "sp_size", 1) != 1:
                raise ValueError(
                    "Published NanoKV codec requires single-rank inference"
                )
            if any("lora_" in name for name, _ in self.model.named_parameters()):
                raise ValueError("NanoKV ablations cannot include LoRA")
            if not self.identity:
                if self.codec_path is None:
                    raise ValueError(
                        "Compression requires the verified published codec"
                    )
                self.packed = validate_codec(self.codec_path)
                if len(self.model.blocks) != 40:
                    raise ValueError(
                        "Published NanoKV requires forty native transformer layers"
                    )
            self._helpers()
            for block in self.model.blocks:
                layer = block.self_attn
                native = layer.forward
                if (
                    "forward" in layer.__dict__
                    or hasattr(layer, "memory_reader")
                    or layer._forward_hooks
                    or layer._forward_pre_hooks
                ):
                    raise ValueError(
                        "NanoKV requires isolated unwrapped self-attention without readers/retrieval"
                    )
                if (
                    not inspect.ismethod(native)
                    or native.__func__.__module__ != "wan.modules.model_fast"
                    or not inspect.isfunction(native.__func__.__globals__["attention"])
                    or layer.local_attn_size != self.capacity_frames
                    or layer.sink_size != 6
                ):
                    raise ValueError(
                        "NanoKV requires native LingBot v2 attention and matching cache/sinks"
                    )
                attention = native.__func__.__globals__["attention"]
                if (
                    attention.__module__ != "worldsync.matrix_attention"
                    or attention.__name__ != "sdpa_attention"
                ):
                    raise ValueError(
                        "NanoKV proof requires the existing unwrapped SDPA backend"
                    )
                self.attention = attention
                if not self.identity and (
                    layer.num_heads != 40 or layer.head_dim != 128
                ):
                    raise ValueError(
                        "NanoKV codec head dimensions differ from the model"
                    )
                self.bindings.append((layer, native))

                def forward(bound, *args, _native=native, **kwargs):
                    return self._forward(bound, _native, *args, **kwargs)

                layer.forward = types.MethodType(forward, layer)
            name = "_initialize_self_kv_cache"
            self.allocator_binding = (
                name in self.pipeline.__dict__,
                self.pipeline.__dict__.get(name),
            )
            original = getattr(self.pipeline, name)

            def allocate(bound, *args, **kwargs):
                return self._allocate(original, *args, **kwargs)

            setattr(self.pipeline, name, types.MethodType(allocate, self.pipeline))
            self.handles.append(
                self.model.register_forward_pre_hook(self._before, with_kwargs=True)
            )
            self.handles.append(
                self.model.register_forward_hook(
                    self._after, with_kwargs=True, always_call=True
                )
            )
            self.active = True
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        for handle in self.handles:
            handle.remove()
        for layer, _ in self.bindings:
            del layer.forward
        if self.allocator_binding is not None:
            existed, value = self.allocator_binding
            if existed:
                self.pipeline._initialize_self_kv_cache = value
            else:
                del self.pipeline._initialize_self_kv_cache
        for cache in self.caches:
            cache["codec_segments"].clear()
            cache["codec_commit"] = False
        self.caches.clear()
        self.operator_cache.clear()
        self.packed = None
        for name in list(sys.modules):
            if name == self.namespace or name.startswith(self.namespace + "."):
                del sys.modules[name]
        self.handles.clear()
        self.bindings.clear()
        self.clean = self.active = False
        self.closed = True

    def metadata(self):
        return {
            "method": "NanoKV identity cache"
            if self.identity
            else "NanoKV compression-only",
            "identity": self.identity,
            "capacity_frames": self.capacity_frames,
            "sink_frames": 6,
            "codec_sha256": None if self.identity else CODEC_SHA256,
            "source_hashes": self.source_hashes,
            "payload_peak_bytes": self.payload_peak_bytes,
            "operator_bytes": self.operator_bytes,
            "closed": self.closed,
            "trace": self.trace,
            "benchmark_eligible": False,
        }

    def __exit__(self, *_):
        self.close()
