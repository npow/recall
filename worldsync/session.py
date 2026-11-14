"""Headless ABot session used by the adaptive player."""
import importlib
from pathlib import Path
import random
import sys

import numpy as np
import torch

from worldsync.memory import MemoryConfig, enable_memory


class ABotSession:
    def __init__(self, repo: Path, height=704, width=1280):
        if height % 32 or width % 32:
            raise ValueError("resolution must be divisible by 32")
        self.repo, self.height, self.width = repo.resolve(), height, width
        sys.path.insert(0, str(self.repo))
        from omegaconf import OmegaConf
        from pipeline import CausalInferencePipeline
        from utils.wan_wrapper import create_vae_from_config
        importlib.import_module("wan.modules.attention").DEFAULT_ATTN_BACKEND = "sdpa"
        torch.set_grad_enabled(False)
        self.device = torch.device("cuda")
        config = OmegaConf.merge(OmegaConf.load(self.repo / "configs/default_config.yaml"), OmegaConf.load(self.repo / "configs/long_forcing_dmd.yaml"))
        config.image_or_video_shape[3] = height // 16
        config.image_or_video_shape[4] = width // 16
        config.model_kwargs.model_name = str(self.repo / config.model_kwargs.model_name)
        for name in ("tokenizer_path", "encoder_pth_path"):
            config.text_encoder_kwargs[name] = str(self.repo / config.text_encoder_kwargs[name])
        config.vae_kwargs.pretrained_path = str(self.repo / config.vae_kwargs.pretrained_path)
        config.taew2_2_checkpoint = str(self.repo / config.taew2_2_checkpoint)
        vae = create_vae_from_config(config)
        self.pipeline = CausalInferencePipeline(config, device=self.device, vae=vae).to(dtype=torch.bfloat16)
        for module in (self.pipeline.generator, self.pipeline.text_encoder, self.pipeline.vae, self.pipeline.encoder):
            if module is not None:
                module.to(self.device)
        self.pipeline.torch_dtype = torch.bfloat16
        self.config = config

    def reset(self, *, image: Path, caption: str, seed: int, mode: str, ref_dir: Path):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        pipeline = self.pipeline
        for module in pipeline.generator.model.modules():
            if hasattr(module, "_worldsync_config"):
                del module._worldsync_config
        if mode != "stock":
            enable_memory(pipeline.generator.model, MemoryConfig(mode=mode))
        pipeline.set_prompts([caption], device=self.device)
        pipeline.set_ref_latent_mask_from_exists_paths(str(ref_dir), device=self.device)
        pipeline.set_first_frame_latent(str(image), height=self.height, width=self.width, device=self.device)
        pipeline._clear_generation_caches()
        pipeline.vae.model.clear_cache()
        pipeline.reset_stream(batch_size=1, dtype=torch.bfloat16, device=self.device)

    def step(self, keys: list[str]):
        pipeline = self.pipeline
        count = int(self.config.num_frame_per_block)
        pipeline.set_act({key: True for key in keys}, height=self.height, width=self.width, num_frames=count, device=self.device)
        noise = torch.randn(1, count, 48, self.height // 16, self.width // 16, device=self.device, dtype=torch.bfloat16)
        latent = pipeline.generate_next_block(noise)

        class Capture:
            def __init__(self):
                self.frames = []
            def append_data(self, frame):
                self.frames.append(frame)

        capture = Capture()
        pipeline.decode_block_and_write(latent, capture)
        if not capture.frames:
            raise RuntimeError("decoder returned no frames")
        return capture.frames
