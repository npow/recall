"""Clip encoding and datasets for memory-keyframe LoRA training.

Two sources, one interface (`worldsync.lingbot_memory_train.Clip`):

* manifest.jsonl (the sibling data fork's output, /data/manifest.jsonl on the
  `worldsync-train-data` volume): one entry per training example
  {video, start, end, poses, keyframe_time, revisit_time, overlap, [prompt, intrinsics]}.
  `video` is an mp4; `start`/`end` are video frame indices; `poses` is an .npy of
  OpenCV camera-to-world matrices per video frame; times are seconds since `start`
  (16 FPS, 4 video frames per latent frame).
* synthetic revisit clips built from still images (pure pan out and back). They
  prove the training step and measure whether the adapter learns to read memory;
  they are not evidence about real-world revisit quality.
"""
import json
import math
from pathlib import Path

import numpy as np

FPS, LATENT_STRIDE = 16, 4
HEIGHT, WIDTH = 480, 832


def latent_index(seconds):
    """Seconds since clip start -> chunk-aligned latent frame index."""
    return int(round(seconds * FPS / LATENT_STRIDE)) // 4 * 4


def plucker_embeddings(poses, intrinsics, device):
    """Framewise-relative camera rays for latent-frame poses [N,4,4] -> [1,384,N,60,104]."""
    import torch
    from einops import rearrange
    from wan.utils.cam_utils import compute_relative_poses, get_plucker_embeddings, get_Ks_transformed
    from worldsync.lingbot_controls import fp32_camera_rays
    n = len(poses)
    relative = compute_relative_poses(torch.from_numpy(np.asarray(poses)).float(), framewise=True).to(device)
    ks = get_Ks_transformed(torch.from_numpy(np.asarray(intrinsics, dtype=np.float32)[None]).float(),
        height_org=480, width_org=832, height_resize=HEIGHT, width_resize=WIDTH,
        height_final=HEIGHT, width_final=WIDTH)[0].repeat(n, 1).to(device)
    rays = fp32_camera_rays(get_plucker_embeddings, relative, ks, HEIGHT, WIDTH)
    rays = rearrange(rays, "f (h a) (w b) c -> (f h w) (c a b)", a=8, b=8)[None]
    return rearrange(rays, "b (f h w) c -> b c f h w", f=n, h=HEIGHT // 8, w=WIDTH // 8)


def encode_clip(pipeline, frames, poses, intrinsics, prompt_embedding, revisit_chunk, memory_start, wrong_start=None, name=""):
    """frames: uint8 [4(N-1)+1, 480, 832, 3]; poses: [N,4,4] per latent frame."""
    import torch
    from worldsync.lingbot_memory_train import Clip
    device = pipeline.device
    video = torch.from_numpy(frames).to(device).permute(3, 0, 1, 2).float() / 127.5 - 1  # [3,F,H,W]
    n = (video.shape[1] - 1) // LATENT_STRIDE + 1
    if len(poses) != n:
        raise ValueError(f"need {n} latent-frame poses, got {len(poses)}")
    with torch.no_grad():
        latent = pipeline.vae.encode([video])[0]
        # I2V conditioning exactly as pipeline.generate builds it: mask 1 on frame 0, VAE of [frame0, zeros].
        zeros = torch.zeros(3, video.shape[1] - 1, HEIGHT, WIDTH, device=device)
        y_latent = pipeline.vae.encode([torch.cat([video[:, :1], zeros], dim=1)])[0]
        mask = torch.zeros(4, n, latent.shape[2], latent.shape[3], device=device)
        mask[:, 0] = 1
        y = torch.cat([mask, y_latent])
        plucker = plucker_embeddings(poses, intrinsics, device).to(pipeline.param_dtype)
    return Clip(latent.to(torch.bfloat16), y.to(torch.bfloat16), plucker, prompt_embedding.to(device),
                revisit_chunk, memory_start, wrong_start, name)


def pan_profile(f):
    """Pan offset (fraction of the pan range) per latent frame for the synthetic revisit clip.

    Sink frames 0-5 sit at 0; the camera pans out to 1.0 by frame 11 (memory chunk 8-11 covers
    offsets 0.4-1.0), returns to 0.3 by frame 21 (previous chunk 20-23 sits near 0.3), then pans
    back to 0.8 over the target chunk 24-27 (offsets 0.425-0.8). The target's right-hand content
    is absent from the sink and the previous chunk, so only the memory chunk supplies it.
    """
    if f < 6:
        return 0.0
    if f <= 11:
        return (f - 6) / 5
    if f <= 13:
        return 1.0
    if f <= 21:
        return 1.0 - 0.7 * (f - 13) / 8
    if f <= 23:
        return 0.3
    return 0.3 + 0.5 * (f - 23) / 4


def synthetic_pan_clip(image_path, intrinsics, latent_frames=28):
    """Pan out and back over a mirror-tiled canvas: frames, per-latent poses, offset profile.

    The pan is a pure rotation (a crop window sliding over a wider canvas), so
    yaw = atan(dx / fx) is consistent with the camera conditioning. Mirror tiling
    keeps seams continuous but makes content differ across tiles.
    """
    from PIL import Image
    fx = float(intrinsics[0]) * WIDTH / 832.0
    image = Image.open(image_path).convert("RGB")
    scale = 640 / image.height
    tile = np.asarray(image.resize((round(image.width * scale), 640), Image.BICUBIC))
    canvas = np.concatenate([tile, tile[:, ::-1], tile, tile[:, ::-1]], axis=1)
    span = min(canvas.shape[1] - WIDTH, 1664)
    top = (640 - HEIGHT) // 2
    total = 4 * (latent_frames - 1) + 1
    frames = []
    for video_frame in range(total):
        f = video_frame / 4
        lo, hi = int(math.floor(f)), min(int(math.ceil(f)), latent_frames - 1)
        g = pan_profile(lo) + (pan_profile(hi) - pan_profile(lo)) * (f - lo)
        x = int(round(g * span))
        frames.append(canvas[top:top + HEIGHT, x:x + WIDTH])
    poses = []
    for f in range(latent_frames):
        yaw = math.atan(pan_profile(f) * span / fx)
        c, s = math.cos(yaw), math.sin(yaw)
        pose = np.eye(4)
        pose[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        poses.append(pose)
    return np.stack(frames), np.stack(poses), [pan_profile(f) for f in range(latent_frames)]


def normalize_entry(entry):
    """Accept both the trainer schema and the Sekai builder schema (times in seconds since window start)."""
    entry = dict(entry)
    if "keyframe_time" not in entry and "keyframe_frame" in entry:
        entry["keyframe_time"] = (entry["keyframe_frame"] - entry["start"]) / FPS
        entry["revisit_time"] = (entry["revisit_frame"] - entry["start"]) / FPS
    if "overlap" not in entry and "overlap_score" in entry:
        entry["overlap"] = entry["overlap_score"]
    return entry


def read_manifest(path, content_ok_only=False):
    entries = [normalize_entry(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]
    required = {"video", "start", "end", "poses", "keyframe_time", "revisit_time", "overlap"}
    for entry in entries:
        missing = required - set(entry)
        if missing:
            raise ValueError(f"manifest entry missing {sorted(missing)}")
    if content_ok_only:
        entries = [e for e in entries if e.get("content_ok")]
    # Keep only pairs whose chunk-aligned indices satisfy Clip's window constraints (sink 6, chunk 4).
    def valid(e):
        k, r = latent_index(e["keyframe_time"]), latent_index(e["revisit_time"])
        return 8 <= k and k + 4 <= r - 4 and r - 8 >= 6
    return [e for e in entries if valid(e)]


def split_manifest(entries, heldout_count, max_train=None):
    """Hold out whole source groups (videos) so train and held-out never share a source."""
    groups = sorted({e.get("source_group", e["video"]) for e in entries})
    held_groups, seen = set(), 0
    for group in groups[::-1]:
        if seen >= heldout_count:
            break
        held_groups.add(group)
        seen += sum(1 for e in entries if e.get("source_group", e["video"]) == group)
    held = [e for e in entries if e.get("source_group", e["video"]) in held_groups]
    train = [e for e in entries if e.get("source_group", e["video"]) not in held_groups]
    if max_train:
        train = train[:max_train]
    return train, held[:heldout_count]


def clip_from_entry(pipeline, entry, prompt_embedding, intrinsics, data_root="/data"):
    """Decode one manifest entry into an encoded Clip (frames start..end, 4(N-1)+1 of them)."""
    import imageio.v3 as iio
    root = Path(data_root)
    video = np.stack([frame for frame in iio.imiter(root / entry["video"])][entry["start"]:entry["end"]])
    count = (len(video) - 1) // LATENT_STRIDE * LATENT_STRIDE + 1
    video = video[:count]
    if video.shape[1:3] != (HEIGHT, WIDTH):
        import cv2
        video = np.stack([cv2.resize(f, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA) for f in video])
    poses = np.load(root / entry["poses"])[entry["start"]:entry["start"] + count][::LATENT_STRIDE]
    clip_intrinsics = entry.get("intrinsics", intrinsics)
    if isinstance(clip_intrinsics, str):
        clip_intrinsics = np.load(root / clip_intrinsics)[entry["start"]]
    return encode_clip(pipeline, video, poses, clip_intrinsics, prompt_embedding,
        revisit_chunk=latent_index(entry["revisit_time"]), memory_start=latent_index(entry["keyframe_time"]),
        wrong_start=entry.get("wrong_keyframe_start"), name=entry["video"])
