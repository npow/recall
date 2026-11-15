"""Modal entry points for the Shadow Canvas streaming session (forkable LingBot branches).

Uses the team's weights volume layout (``scripts/modal_lingbot.py``): the 14B
causal-fast checkpoint at ``/weights/14b-causal-fast``.

    modal run [--profile <workspace>] scripts/modal_shadow_canvas.py::download_weights
    modal run [--profile <workspace>] scripts/modal_shadow_canvas.py::validate_streaming
    modal run [--profile <workspace>] scripts/modal_shadow_canvas.py::smoke_turn
"""
import json
from pathlib import Path
import sys
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), "/root"]  # repo root locally; mounted package root in the container
from worldsync.upstream import LINGBOT_REPO, LINGBOT_REVISION  # noqa: E402

APP_NAME = "worldsync-shadow-canvas"
REPO_DIR = "/opt/lingbot"
WEIGHTS = Path("/weights")
OUTPUTS = Path("/outputs")
CKPT_14B = WEIGHTS / "14b-causal-fast"
CKPT_1P3B = WEIGHTS / "1.3b-causal-fast"
WEIGHT_REVISION_14B = "5c33dd40b213598c418fd25bff30fdbd23fd38a7"  # reproduction/lingbot-weights.json
LAKE_PROMPT = ("A serene lakeside scene with a lone tree standing in calm water, surrounded by distant "
               "snow-capped mountains under a bright blue sky with drifting white clouds — gentle ripples "
               "reflect the tree and sky, creating a tranquil, meditative atmosphere.")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install("torch==2.6.0", "torchvision==0.21.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "diffusers==0.33.1", "transformers==4.51.3", "tokenizers==0.21.1", "accelerate==1.6.0",
        "easydict==1.13", "ftfy==6.3.1", "regex==2024.11.6", "imageio==2.37.0", "imageio-ffmpeg==0.6.0",
        "numpy==1.26.4", "scipy==1.15.2", "safetensors==0.5.3", "huggingface_hub[hf_transfer]==0.30.2",
        "einops==0.8.1", "tqdm==4.67.1", "Pillow==11.2.1", "sentencepiece==0.2.0", "protobuf==5.29.4")
    .run_commands(f"git clone {LINGBOT_REPO} {REPO_DIR} && git -C {REPO_DIR} checkout {LINGBOT_REVISION}")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(ROOT / "worldsync", "/root/worldsync",
                   ignore=lambda path: "__pycache__" in Path(path).parts)
)
weights = modal.Volume.from_name("worldsync-lingbot-weights", create_if_missing=True)
outputs = modal.Volume.from_name("worldsync-lingbot-results", create_if_missing=True)
app = modal.App(APP_NAME, image=image)
# 14B bf16 plus T5 and one KV cache peaked near 80 GB in the team's runs; T5 loads via CPU RAM.
GPU_KWARGS = dict(gpu="H200", volumes={WEIGHTS: weights, OUTPUTS: outputs}, timeout=4 * 3600, memory=131072, cpu=8)


@app.function(volumes={WEIGHTS: weights}, timeout=4 * 3600)
def download_weights(variant: str = "14b"):
    """Fetch weights into the team's layout; existing complete files are skipped."""
    from huggingface_hub import snapshot_download
    if variant == "14b":
        snapshot_download("robbyant/lingbot-world-v2-14b-causal-fast", revision=WEIGHT_REVISION_14B,
                          local_dir=CKPT_14B)
    elif variant == "1.3b":  # development only; shares T5/VAE/tokenizer with the 14B release
        snapshot_download("robbyant/lingbot-world-v2-1.3b-causal-fast", local_dir=CKPT_1P3B,
                          allow_patterns=["*.safetensors", "*.json"])
    else:
        raise ValueError("variant must be 14b or 1.3b")
    weights.commit()
    return sorted(str(p.relative_to(WEIGHTS)) for p in WEIGHTS.rglob("*") if p.is_file())


def write_video(path: Path, frames, fps: int = 16):
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)


def session(variant: str = "14b"):
    from worldsync.lingbot import LingBotSession
    if variant == "14b":
        return LingBotSession(str(CKPT_14B), None, REPO_DIR, task="i2v-A14B")
    return LingBotSession(str(CKPT_1P3B), str(CKPT_14B), REPO_DIR, task="i2v-1.3B")


@app.function(**GPU_KWARGS)
def validate_streaming(frame_num: int = 81, seed: int = 42, variant: str = "14b"):
    """Streaming chunk-by-chunk output must match upstream's offline generate()."""
    import numpy as np
    from PIL import Image
    import torch
    from worldsync.lingbot import to_uint8
    sess = session(variant)
    from wan.utils.cam_utils import compute_relative_poses, interpolate_camera_poses
    example = Path(REPO_DIR) / "examples/03"
    img = Image.open(example / "image.jpg").convert("RGB")
    shift = sess.cfg.sample_shift

    start = time.perf_counter()
    offline = sess.pipe.generate(LAKE_PROMPT, img, action_path=str(example), chunk_size=4, max_area=480 * 832,
                                 frame_num=frame_num, shift=shift, seed=seed, offload_model=False)
    offline_seconds = time.perf_counter() - start

    # Reproduce upstream's pose preprocessing, then stream the same relative poses.
    c2ws = np.load(example / "poses.npy")
    frames = min(((frame_num - 1) // 4) * 4 + 1, ((len(c2ws) - 1) // 4) * 4 + 1)
    c2ws = c2ws[:frames]
    latent = (frames - 1) // 4 + 1
    latent -= latent % 4
    infer = interpolate_camera_poses(np.linspace(0, frames - 1, frames), c2ws[:, :3, :3], c2ws[:, :3, 3],
                                     np.linspace(0, frames - 1, latent))
    relative = compute_relative_poses(infer, framewise=True)
    intrinsics = np.load(example / "intrinsics.npy")[0]

    start = time.perf_counter()
    main = sess.reset(img, LAKE_PROMPT, seed, latent_frames=latent, shift=shift, intrinsics=intrinsics)
    reset_seconds = time.perf_counter() - start
    decoder, chunks, chunk_seconds = sess.decoder(), [], []
    for index in range(latent // 4):
        tick = time.perf_counter()
        sess.step(main, relative=relative[index * 4:(index + 1) * 4])
        chunks.append(decoder.decode(main.latents[-1]))
        torch.cuda.synchronize()
        chunk_seconds.append(time.perf_counter() - tick)
    streamed = torch.cat(chunks, dim=1)

    diff = (streamed - offline).abs()
    report = {
        "lingbot_revision": LINGBOT_REVISION, "variant": variant, "frame_num": frame_num, "seed": seed,
        "shape": list(streamed.shape), "offline_shape": list(offline.shape),
        "max_abs_diff": diff.max().item(), "mean_abs_diff": diff.mean().item(),
        "bitwise_equal": bool(torch.equal(streamed, offline)),
        "offline_seconds": offline_seconds, "reset_seconds": reset_seconds, "chunk_seconds": chunk_seconds,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9, "gpu": torch.cuda.get_device_name(),
    }
    out = OUTPUTS / "shadow-canvas" / f"validate-streaming-{variant}"
    write_video(out / "offline.mp4", to_uint8(offline))
    write_video(out / "streamed.mp4", to_uint8(streamed))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    outputs.commit()
    return report


@app.function(**GPU_KWARGS)
def smoke_turn(actions: str = "hold(Left,1800ms); wait(900ms); hold(Right,1800ms); wait(900ms)", seed: int = 42,
               variant: str = "14b"):
    """Drive the streaming session with PlayWorld-style actions on an upstream example."""
    import numpy as np
    from PIL import Image
    import torch
    from worldsync.lingbot import to_uint8
    from worldsync.poses import CameraRig, actions_to_latent_steps, parse_actions
    sess = session(variant)
    img = Image.open(Path(REPO_DIR) / "examples/03/image.jpg").convert("RGB")

    rig = CameraRig()
    poses = [rig.pose.copy()] + [rig.advance(keys, seconds) for keys, seconds in
                                 actions_to_latent_steps(parse_actions(actions))]
    while len(poses) % 4:
        poses.append(rig.advance(()))
    main = sess.reset(img, LAKE_PROMPT, seed, latent_frames=len(poses))
    decoder, video, chunk_seconds = sess.decoder(), [], []
    for index in range(len(poses) // 4):
        tick = time.perf_counter()
        sess.step(main, np.stack(poses[index * 4:(index + 1) * 4]))
        video.append(decoder.decode(main.latents[-1]))
        torch.cuda.synchronize()
        chunk_seconds.append(time.perf_counter() - tick)
    frames = to_uint8(torch.cat(video, dim=1))
    out = OUTPUTS / "shadow-canvas" / f"smoke-turn-{variant}"
    write_video(out / "rollout.mp4", frames)
    Image.fromarray(frames[0]).save(out / "first.png")
    Image.fromarray(frames[len(frames) // 2]).save(out / "middle.png")
    Image.fromarray(frames[-1]).save(out / "last.png")
    report = {"variant": variant, "actions": actions, "latent_frames": len(poses), "video_frames": len(frames),
              "chunk_seconds": chunk_seconds, "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    outputs.commit()
    return report
