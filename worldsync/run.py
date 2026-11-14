"""Paired open-loop rollout diagnostic; this is not the PlayWorld score."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from worldsync.memory import MemoryConfig, enable_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("external/ABot-World"))
    parser.add_argument("--experiment", type=Path, default=Path("experiments/pan_return.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/pan_return"))
    parser.add_argument("--modes", nargs="+", choices=["stock", "window", "landmark"], default=["stock", "window", "landmark"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--landmarks", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--max-blocks", type=int, help="smoke-test truncation; does not complete the return task")
    args = parser.parse_args()
    root = Path.cwd()
    repo, output = args.repo.resolve(), args.output.resolve()
    spec = json.loads(args.experiment.read_text())
    if args.height % 32 or args.width % 32:
        parser.error("height and width must be divisible by 32")
    if args.max_blocks is not None and args.max_blocks < 1:
        parser.error("--max-blocks must be positive")
    image = (repo / spec["image"]).resolve()
    if not image.is_file():
        parser.error(f"missing initial image: {image}")
    if "from worldsync.memory import memory_view" not in (repo / "wan/modules/causal_model.py").read_text():
        parser.error("apply patches/abot-world-memory.patch first")
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    import importlib
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    from pipeline import CausalInferencePipeline
    from utils.wan_wrapper import create_vae_from_config

    importlib.import_module("wan.modules.attention").DEFAULT_ATTN_BACKEND = "sdpa"

    if not torch.cuda.is_available():
        raise RuntimeError("ABot rollout requires a CUDA GPU")
    torch.set_grad_enabled(False)
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/long_forcing_dmd.yaml"))
    config.image_or_video_shape[3] = args.height // 16
    config.image_or_video_shape[4] = args.width // 16
    # Unquantized weights and SDPA for every mode; no architecture-specific kernels.
    device = torch.device("cuda")
    vae = create_vae_from_config(config)
    pipeline = CausalInferencePipeline(config, device=device, vae=vae).to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device)
    pipeline.generator.to(device)
    pipeline.vae.to(device)
    if pipeline.encoder is not None:
        pipeline.encoder.to(device)
    pipeline.torch_dtype = torch.bfloat16
    pipeline.set_prompts([spec["prompt"]], device=device)
    pipeline.set_ref_latent_mask_from_exists_paths(str(output / "unused_refs"), device=device)
    pipeline.set_first_frame_latent(str(image), height=args.height, width=args.width, device=device)
    blocks = [segment["keys"] for segment in spec["segments"] for _ in range(segment["blocks"])]
    planned_blocks = len(blocks)
    if args.max_blocks:
        blocks = blocks[:args.max_blocks]
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    model_revision_path = repo / "checkpoints/ABot-World-0-5B-LF/.cache/huggingface/download/config.json.metadata"
    model_revision = model_revision_path.read_text().splitlines()[0] if model_revision_path.exists() else "unknown-local-checkpoint"
    initial_hash = hashlib.sha256(image.read_bytes()).hexdigest()
    code_hash = hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name("memory.py").read_bytes()).hexdigest()

    for seed in args.seeds:
        for mode in args.modes:
            run_dir = output / f"seed-{seed}" / mode
            run_dir.mkdir(parents=True, exist_ok=False)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            for layer in pipeline.generator.model.modules():
                if hasattr(layer, "_worldsync_config"):
                    del layer._worldsync_config
            if mode != "stock":
                enable_memory(pipeline.generator.model, MemoryConfig(mode=mode, landmark_frames=args.landmarks, novelty_threshold=args.threshold))
            pipeline._clear_generation_caches()
            pipeline.vae.model.clear_cache()
            pipeline.reset_stream(batch_size=1, dtype=torch.bfloat16, device=device)
            torch.cuda.reset_peak_memory_stats()
            result = {"status": "running", "mode": mode, "seed": seed,
                      "upstream_commit": source_commit, "model_revision": model_revision, "torch": torch.__version__,
                      "gpu": torch.cuda.get_device_name(), "height": args.height, "width": args.width,
                      "attention_backend": "sdpa", "quantization": None,
                      "experiment": spec, "image_sha256": initial_hash,
                      "experiment_code_sha256": code_hash,
                      "planned_blocks": planned_blocks, "executed_blocks": len(blocks),
                      "complete_action_sequence": len(blocks) == planned_blocks,
                      "landmark_frames": args.landmarks, "novelty_threshold": args.threshold,
                      "evaluation": "open-loop diagnostic, NOT PlayWorld", "blocks": []}
            result_path = run_dir / "result.json"
            result_path.write_text(json.dumps(result, indent=2))
            start = time.perf_counter()
            try:
                with imageio.get_writer(str(run_dir / "rollout.mp4"), fps=spec["fps"], codec="libx264", macro_block_size=1) as writer:
                    class Capture:
                        def __init__(self):
                            self.frames = []
                        def append_data(self, frame):
                            self.frames.append(frame)
                            writer.append_data(frame)

                    for index, keys in enumerate(blocks):
                        tick = time.perf_counter()
                        num_frames = int(config.num_frame_per_block)
                        pipeline.set_act({key: True for key in keys}, height=args.height, width=args.width, num_frames=num_frames, device=device)
                        noise = torch.randn([1, num_frames, 48, args.height // 16, args.width // 16], device=device, dtype=torch.bfloat16)
                        latent = pipeline.generate_next_block(noise)
                        capture = Capture()
                        pipeline.decode_block_and_write(latent, capture)
                        if not capture.frames:
                            raise RuntimeError("decoder returned no frames")
                        Image.fromarray(capture.frames[-1]).save(run_dir / f"block-{index:03d}.png")
                        if index == 0:
                            Image.fromarray(capture.frames[0]).save(run_dir / "initial.png")
                        Image.fromarray(capture.frames[-1]).save(run_dir / "final.png")
                        torch.cuda.synchronize()
                        stats = pipeline.kv_cache1[0].get("_worldsync")
                        record = {"index": index, "keys": keys, "seconds": time.perf_counter() - tick,
                                  "decoded_frames": len(capture.frames), "memory": stats.stats() if stats else None}
                        result["blocks"].append(record)
                        result_path.write_text(json.dumps(result, indent=2))
                        print(f"{mode} seed={seed} block={index + 1}/{len(blocks)} seconds={record['seconds']:.2f}", flush=True)
                result["status"] = "completed"
            except Exception as exc:
                result["status"] = "failed"
                result["error"] = repr(exc)
                raise
            finally:
                result["elapsed_seconds"] = time.perf_counter() - start
                result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                result_path.write_text(json.dumps(result, indent=2))
    os.chdir(root)


if __name__ == "__main__":
    main()
