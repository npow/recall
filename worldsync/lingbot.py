"""Streaming LingBot-World2 causal-fast inference with forkable branches.

Mirrors ``WanI2VCausal._generate_causal_fast`` (pinned upstream revision) chunk by
chunk, reusing the team's verified pieces: ``lingbot_native`` loading,
``matrix_attention`` SDPA and ``lingbot_vae`` streaming decode. Unlike the
hook-based ``lingbot_rollout``, the loop is owned here so branches can fork. A branch owns a self-attention KV cache, an RNG and a camera pose. Branches
forked from one another advance at identical absolute frame positions, so cached
frames can be copied between them without re-indexing. Requires a CUDA GPU and
the upstream repository on ``sys.path``.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange

from worldsync.poses import DEFAULT_INTRINSICS, STEP_PER_LATENT, relative_poses, scaled_intrinsics

DEFAULT_TIMESTEPS_INDEX = (0, 250, 500, 750)


def import_lingbot(repo_dir: str):
    """Import upstream with the same SDPA attention bridge as ``lingbot_native``."""
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    import wan
    from wan.modules import model, model_fast
    from worldsync.matrix_attention import sdpa_attention
    model_fast.flash_attention = model_fast.attention = model.flash_attention = sdpa_attention
    return wan


def use_strict_low_memory_loading(upstream) -> None:
    """Load 14B shards with ``from_pretrained`` as ``lingbot_native`` does.

    Upstream's fallback builds a float32 model before loading, transiently
    duplicating the 74 GB checkpoint. Any key mismatch is an error.
    """
    def strict_load(cls, checkpoint_dir, subfolder, config, torch_dtype, extra=None):
        model, info = cls.from_pretrained(str(Path(checkpoint_dir) / subfolder), torch_dtype=torch_dtype,
                                          low_cpu_mem_usage=True, local_files_only=True,
                                          output_loading_info=True, **(extra or {}))
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"checkpoint does not match the official model: {info}")
        return model
    upstream.load_dit_model = strict_load


@dataclass
class Branch:
    name: str
    kv: list[dict]
    generator: torch.Generator
    chunk: int = 0
    pose: np.ndarray | None = None
    latents: list[torch.Tensor] = field(default_factory=list)


def to_uint8(video: torch.Tensor) -> np.ndarray:
    """[3, N, H, W] in [-1, 1] to [N, H, W, 3] uint8, as upstream ``save_video``."""
    return ((video.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()


class LingBotSession:

    def __init__(self, ckpt_dir: str, assets_dir: str | None, repo_dir: str, task: str = "i2v-A14B",
                 local_attn_size: int = 18, sink_size: int = 6, chunk_size: int = 4, device_id: int = 0,
                 pack_dir: str = "/tmp/lingbot-runtime-pack"):
        wan = import_lingbot(repo_dir)
        from wan.configs import WAN_CONFIGS
        import wan.image2video as upstream
        self.cfg = WAN_CONFIGS[task]
        if (Path(ckpt_dir) / "transformers").is_dir():  # 14B layout, as in lingbot_native
            from worldsync.lingbot_native import runtime_pack
            ckpt_dir = str(runtime_pack(Path(ckpt_dir), Path(pack_dir)))
            use_strict_low_memory_loading(upstream)
        self.pipe = wan.WanI2VCausal(
            config=self.cfg, checkpoint_dir=ckpt_dir, device_id=device_id, local_attn_size=local_attn_size,
            sink_size=sink_size, infer_mode="causal_fast", assets_dir=assets_dir, convert_model_dtype=True)
        self.model, self.device, self.chunk_size = self.pipe.model, self.pipe.device, chunk_size
        self.local_attn_size, self.sink_size = local_attn_size, sink_size

    # ---- episode setup -------------------------------------------------

    @torch.no_grad()
    def reset(self, image, prompt: str, seed: int, latent_frames: int, max_area: int = 480 * 832,
              shift: float | None = None, timesteps_index=DEFAULT_TIMESTEPS_INDEX,
              trans_scale: float = STEP_PER_LATENT, intrinsics: np.ndarray = DEFAULT_INTRINSICS) -> Branch:
        """Prepare an episode of at most ``latent_frames`` and return its main branch."""
        pipe, cfg, cs = self.pipe, self.cfg, self.chunk_size
        img = TF.to_tensor(image).sub_(0.5).div_(0.5).to(self.device)
        h, w = img.shape[1:]
        lat_h = round(np.sqrt(max_area * h / w) // cfg.vae_stride[1] // cfg.patch_size[1] * cfg.patch_size[1])
        lat_w = round(np.sqrt(max_area * w / h) // cfg.vae_stride[2] // cfg.patch_size[2] * cfg.patch_size[2])
        self.lat_h, self.lat_w = lat_h, lat_w
        self.h, self.w = lat_h * cfg.vae_stride[1], lat_w * cfg.vae_stride[2]
        self.lat_f = latent_frames - latent_frames % cs
        frames = (self.lat_f - 1) * cfg.vae_stride[0] + 1
        self.frame_seqlen = lat_h * lat_w // (cfg.patch_size[1] * cfg.patch_size[2])
        self.max_seq_len = cs * self.frame_seqlen
        self.kv_size = self.frame_seqlen * self.local_attn_size
        self.trans_scale = trans_scale
        self.Ks = torch.tensor(scaled_intrinsics(self.h, self.w, intrinsics), dtype=torch.float32,
                               device=self.device)[None]

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        self.noise = torch.randn(16, self.lat_f, lat_h, lat_w, dtype=torch.float32, generator=generator,
                                 device=self.device)

        msk = torch.ones(1, frames, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.cat([torch.repeat_interleave(msk[:, :1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]

        pipe.scheduler.set_timesteps(pipe.num_train_timesteps, shift=cfg.sample_shift if shift is None else shift)
        self.timesteps = pipe.scheduler.timesteps[list(timesteps_index)]
        self.context = self.encode_prompt(prompt)

        resized = F.interpolate(img[None].cpu(), size=(self.h, self.w), mode="bicubic").transpose(0, 1)
        video = torch.cat([resized, torch.zeros(3, frames - 1, self.h, self.w)], dim=1).to(self.device)
        self.y = torch.cat([msk, pipe.vae.encode([video])[0]])
        del video

        head_dim = self.model.config.dim // self.model.config.num_heads
        heads = self.model.config.num_heads
        self.cross_kv = pipe._initialize_crossattn_cache(
            num_layers=self.model.config.num_layers, shape=[1, 512, heads, head_dim],
            dtype=pipe.pipe_dtype, device=self.device)
        self.cross_initialized = False
        kv = pipe._initialize_self_kv_cache(
            num_layers=self.model.config.num_layers, shape=[1, self.kv_size, heads, head_dim],
            dtype=pipe.pipe_dtype, device=self.device)
        return Branch("main", kv, generator)

    def encode_prompt(self, prompt: str):
        pipe = self.pipe
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if key not in pipe._t5_cache:
            pipe.text_encoder.model.to(self.device)
            pipe._t5_cache[key] = pipe.text_encoder([prompt], self.device)
            pipe.text_encoder.model.cpu()
            torch.cuda.empty_cache()
        return pipe._t5_cache[key]

    # ---- generation ----------------------------------------------------

    def plucker(self, relative: torch.Tensor) -> torch.Tensor:
        from wan.utils.cam_utils import get_plucker_embeddings
        emb = get_plucker_embeddings(relative, self.Ks.repeat(len(relative), 1), self.h, self.w)
        emb = rearrange(emb, "f (h c1) (w c2) c -> (f h w) (c c1 c2)", c1=self.h // self.lat_h,
                        c2=self.w // self.lat_w)[None]
        return rearrange(emb, "b (f h w) c -> b c f h w", f=len(relative), h=self.lat_h,
                         w=self.lat_w).to(self.pipe.param_dtype)

    @contextmanager
    def _inference(self):
        no_sync = getattr(self.model, "no_sync", None)
        with torch.amp.autocast("cuda", dtype=self.pipe.param_dtype), torch.no_grad():
            if no_sync is None:
                yield
            else:
                with no_sync():
                    yield

    def step(self, branch: Branch, c2ws: np.ndarray | None = None, relative: torch.Tensor | None = None) -> torch.Tensor:
        """Generate one chunk on ``branch`` and commit it to its KV cache.

        ``c2ws`` are absolute camera-to-world poses of the chunk's latent frames.
        ``relative`` bypasses pose conversion with LingBot-format relative poses.
        """
        cs, index = self.chunk_size, branch.chunk
        if (index + 1) * cs > self.lat_f:
            raise ValueError(f"episode prepared for {self.lat_f} latent frames")
        if relative is None:
            relative = torch.from_numpy(relative_poses(c2ws, branch.pose, self.trans_scale)).float()
        if len(relative) != cs:
            raise ValueError(f"expected {cs} poses per chunk")
        frames = slice(index * cs, (index + 1) * cs)
        if branch.name == "main":
            latent = self.noise[:, frames]
        else:
            latent = torch.randn(16, cs, self.lat_h, self.lat_w, dtype=torch.float32,
                                 generator=branch.generator, device=self.device)
        kwargs = {
            "context": [self.context[0]], "seq_len": self.max_seq_len, "y": [self.y[:, frames]],
            "dit_cond_dict": {"c2ws_plucker_emb": self.plucker(relative.to(self.device)).chunk(1, dim=0)},
            "kv_cache": branch.kv, "crossattn_cache": self.cross_kv,
            "current_start": index * cs * self.frame_seqlen, "max_attention_size": self.kv_size,
            "frame_seqlen": self.frame_seqlen,
        }
        with self._inference():
            for step, timestep in enumerate(self.timesteps):
                t = torch.stack([timestep]).to(self.device)
                flow = self.model(x=[latent.to(self.device)], t=t, cross_attn_first_call=not self.cross_initialized,
                                  **kwargs)[0]
                self.cross_initialized = True
                x0 = self.pipe._convert_flow_pred_to_x0(flow_pred=flow, xt=latent, timestep=timestep,
                                                        scheduler=self.pipe.scheduler)
                if step < len(self.timesteps) - 1:
                    noise = torch.randn(x0.shape, generator=branch.generator, device=x0.device, dtype=x0.dtype)
                    latent = self.pipe.scheduler.add_noise(x0, noise, self.timesteps[step + 1])
            t0 = torch.stack([self.timesteps[-1] * 0.0]).to(self.device)
            self.model(x=[x0], t=t0, cross_attn_first_call=False, **kwargs)
        branch.chunk += 1
        if c2ws is not None:
            branch.pose = np.asarray(c2ws[-1], dtype=float)
        branch.latents.append(x0)
        return x0

    # ---- branch operations ---------------------------------------------

    def fork(self, branch: Branch, name: str, seed: int) -> Branch:
        """Copy ``branch``'s full simulation state into an independent branch."""
        if name == "main":
            raise ValueError("only the episode's original branch may be named main")
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        kv = [{key: value.clone() for key, value in layer.items()} for layer in branch.kv]
        pose = None if branch.pose is None else branch.pose.copy()
        return Branch(name, kv, generator, branch.chunk, pose)

    def cached_frames(self, branch: Branch) -> int:
        return int(branch.kv[0]["local_end_index"].item()) // self.frame_seqlen

    def insert_frames(self, dst: Branch, src: Branch, slots: list[int], frames: int | None = None) -> None:
        """Overwrite cache frame ``slots`` of ``dst`` with ``src``'s newest frames.

        Slots index frames in the cache (0..sink_size-1 are attention-sink frames).
        Source frames are taken in temporal order from the end of ``src``'s cache.
        """
        frames = len(slots) if frames is None else frames
        if frames != len(slots):
            raise ValueError("one source frame per destination slot")
        valid_dst, valid_src = self.cached_frames(dst), self.cached_frames(src)
        if frames > valid_src or any(not 0 <= s < valid_dst for s in slots):
            raise ValueError("slots and source frames must lie in committed cache regions")
        fs = self.frame_seqlen
        for d_layer, s_layer in zip(dst.kv, src.kv):
            for i, slot in enumerate(slots):
                start = (valid_src - frames + i) * fs
                for key in ("k", "v"):
                    d_layer[key][:, slot * fs:(slot + 1) * fs] = s_layer[key][:, start:start + fs]

    def decoder(self):
        """Native streaming VAE decoder (``lingbot_vae``), verified against native pixels.

        Its feature cache lives on the shared VAE, so create it after ``reset`` (which
        VAE-encodes the conditioning and clears that cache) and keep one per episode.
        """
        from worldsync.lingbot_vae import NativeVaeStream
        return NativeVaeStream(self.pipe.vae)
