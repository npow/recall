"""Run the stock official LingBot-World2 14B checkpoint on the local Thor host.

This validates native generation before integrating the adaptive player. It is
not a PlayWorld score. Only loading, attention backend and output transport are
adapted; weights, sampler and KV-cache policy retain official demo settings.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "1895d300d8ac936401689b26389f51cbd36530eb"
WEIGHT_REVISION = "5c33dd40b213598c418fd25bff30fdbd23fd38a7"


def runtime_pack(checkpoint, output):
    """Expose upstream's root config alongside its shards for low-memory loading.

    All links point at verified original files; the raw HF layout stays intact.
    The official fallback builds a full float32 model before loading it, which
    would transiently duplicate the 74 GB checkpoint on this shared-memory host.
    """
    output.mkdir(parents=True, exist_ok=True)
    for item in checkpoint.iterdir():
        if item.name in {"transformers", ".cache"}:
            continue
        destination = output / item.name
        if not destination.exists():
            destination.symlink_to(item)
    transformers = output / "transformers"
    transformers.mkdir(exist_ok=True)
    for item in (checkpoint / "transformers").iterdir():
        destination = transformers / item.name
        if not destination.exists():
            destination.symlink_to(item)
    config = transformers / "config.json"
    if not config.exists():
        config.symlink_to(checkpoint / "config.json")
    return output


def configure_backend(report, save):
    """Use strict native loading and the verified local attention backend."""
    import torch
    source = ROOT / "external/LingBot-World2"
    sys.path.insert(0, str(source))
    import wan.image2video as upstream
    from wan.modules import model_fast, model as base_model
    from worldsync.matrix_attention import sdpa_attention
    model_fast.flash_attention = sdpa_attention
    model_fast.attention = sdpa_attention
    base_model.flash_attention = sdpa_attention
    def strict_load(cls, checkpoint_dir, subfolder, config, torch_dtype, extra=None):
        model, info = cls.from_pretrained(str(Path(checkpoint_dir) / subfolder),
            torch_dtype=torch_dtype, low_cpu_mem_usage=True, local_files_only=True,
            output_loading_info=True, **(extra or {}))
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"checkpoint does not match the official model: {info}")
        report["checkpoint_loading"] = {key: info.get(key, []) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
        save()
        return model
    upstream.load_dit_model = strict_load
    return upstream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["gc", "if", "insight", "outsight"], default="gc")
    parser.add_argument("--task-id", default="GC007")
    parser.add_argument("--frames", type=int, default=13)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.frames < 13 or ((args.frames - 1) // 4 + 1) % 4 or (args.frames - 1) % 4:
        parser.error("native frames must equal 16*n-3 (13, 29, 45, ...) for four-latent chunks")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "external/LingBot-World2"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if revision != SOURCE_REVISION:
        parser.error("LingBot source revision differs from frozen source")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "wan", "generate.py"], cwd=source, check=True)
    data = ROOT / "external/PlayWorld/data"
    task = next((record for record in json.loads((data / args.split / "data.json").read_text()) if record["task_id"] == args.task_id), None)
    if task is None:
        parser.error("task not in split")
    report = {"model": "LingBot-World2 14B causal-fast", "status": "prepared",
        "scope": "stock native generation with neutral camera, not an adaptive PlayWorld rollout",
        "source_revision": revision, "checkpoint_revision": WEIGHT_REVISION,
        "task_id": args.task_id, "split": args.split, "frames_requested": args.frames,
        "fps": 16, "max_area": 480 * 832, "chunk_size": 4, "local_attn_size": 18, "sink_size": 6,
        "seed": 42, "sampler_timesteps_index": [0, 250, 500, 750], "sample_shift": 10,
        "quantization": "none", "published_score_reproduced": False,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "hardware_substitutions": ["single Thor GPU with BF16 and PyTorch SDPA",
            "root config exposed alongside shards for native low-memory from_pretrained loading"]}
    path = output / "run.json"
    def save():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(path)
    save()
    if args.dry_run:
        print(json.dumps({"status": "prepared", "api_calls": 0, "frames": args.frames}))
        return
    manifest = json.loads((ROOT / "reproduction/lingbot-weights.json").read_text())
    if manifest["revision"] != WEIGHT_REVISION or manifest["variant"] != "14B causal-fast":
        raise ValueError("wrong model inventory")
    raw = source / "checkpoints/lingbot-world-v2-14b-causal-fast"
    for record in manifest["files"]:
        if (raw / record["path"]).stat().st_size != record["bytes"]:
            raise ValueError("incomplete verified checkpoint")
    pack = runtime_pack(raw, source / "checkpoints/runtime-14b-causal-fast")
    import numpy as np
    import torch
    from PIL import Image
    import imageio.v2 as imageio
    sys.path.insert(0, str(source))
    from wan.configs import WAN_CONFIGS
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    upstream = configure_backend(report, save)
    controls = output / "controls"
    controls.mkdir()
    np.save(controls / "poses.npy", np.repeat(np.eye(4, dtype=np.float32)[None], args.frames, axis=0))
    native_intrinsics = np.load(source / "examples/03/intrinsics.npy")[0]
    np.save(controls / "intrinsics.npy", np.repeat(native_intrinsics[None], args.frames, axis=0))
    report["status"] = "running"
    report["environment"] = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    save()
    started = time.monotonic()
    try:
        pipeline = upstream.WanI2VCausal(WAN_CONFIGS["i2v-A14B"], str(pack),
            local_attn_size=18, sink_size=6, infer_mode="causal_fast", convert_model_dtype=True)
        video = pipeline.generate(task["image_caption"], Image.open(data / task["image_path"]).convert("RGB"),
            action_path=str(controls), chunk_size=4, max_area=480 * 832, frame_num=args.frames,
            shift=10, seed=42, offload_model=False)
        if not torch.isfinite(video).all():
            raise ValueError("native output contains non-finite pixels")
        pixels = ((video.permute(1, 2, 3, 0).float() + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
        if len(pixels) != args.frames:
            raise ValueError("native output is incomplete or invalid")
        movie = output / "rollout.mp4"
        imageio.mimwrite(movie, pixels, fps=16, codec="libx264", quality=8, macro_block_size=1)
        if args.frames <= 29:
            np.save(output / "pixels.npy", pixels)
        Image.fromarray(pixels[-1]).save(output / "observation.jpg", quality=95)
        report.update(status="completed", frames=len(pixels), frame_shape=list(pixels.shape[1:]),
            pixel_sha256=hashlib.sha256(pixels).hexdigest(),
            video_sha256=hashlib.sha256(movie.read_bytes()).hexdigest())
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - started
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        save()
    print(json.dumps({key: report[key] for key in ("status", "frames", "elapsed_s", "peak_allocated_bytes")}))


if __name__ == "__main__":
    main()
