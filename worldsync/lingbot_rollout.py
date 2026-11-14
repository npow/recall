"""Adaptive stock LingBot rollout through model I/O hooks; native KV policy retained."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

from worldsync.credentials import load_local_credentials
from worldsync.lingbot_controls import CameraPlan
from worldsync.lingbot_native import ROOT, SOURCE_REVISION, WEIGHT_REVISION, runtime_pack, configure_backend

PROTOCOL_ID = "lingbot-world2-stock-reconstructed-haiku-16fps-v1"


class RolloutComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "external/PlayWorld/data")
    parser.add_argument("--split", choices=["gc", "if", "insight", "outsight"], required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--max-chunks", type=int, default=40)
    parser.add_argument("--fixed-actions", nargs="+", help="Transport/parity diagnostic; not adaptive benchmark evidence")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_chunks <= 60:
        parser.error("chunk budget must be in [1,60]")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "external/LingBot-World2"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if revision != SOURCE_REVISION:
        parser.error("unexpected LingBot source revision")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "wan", "generate.py"], cwd=source, check=True)
    task = next((record for record in json.loads((args.data_root / args.split / "data.json").read_text()) if record["task_id"] == args.task_id), None)
    if task is None:
        parser.error("task not found")
    initial = (args.data_root / task["image_path"]).resolve()
    stationary = args.split == "insight"
    maximum = min(args.max_chunks, 60 if stationary else 40)
    frames = maximum * 16 - 3
    report = {"status": "prepared", "model": "LingBot-World2 14B causal-fast stock", "protocol_id": PROTOCOL_ID,
        "source_revision": revision, "checkpoint_revision": WEIGHT_REVISION, "task_id": args.task_id,
        "split": args.split, "seed": 42, "fps": 16, "max_chunks": maximum, "max_interactions": 40,
        "stationary_wait_only": stationary, "fixed_actions": args.fixed_actions, "quantization": "none",
        "max_area": 480 * 832, "chunk_size": 4, "local_attn_size": 18, "sink_size": 6,
        "sampler_timesteps_index": [0, 250, 500, 750], "sample_shift": 10,
        "actions": [], "chunks": [], "published_score_reproduced": False,
        "agent_interactions": 0,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "deviations": ["Hosted PlayWorld variant/settings not verified; official local 14B causal-fast checkpoint",
            "Reconstructed Haiku reference-guided prompt and camera mapping: 60 deg/s turns, unit-normalized translations",
            "Official 480p demo area and KV settings; single Thor GPU, BF16 and valid-length SDPA backend",
            "Native VAE sequential decoding exposed at chunk boundaries for observations; parity verified separately",
            "Claude diagnostic scoring differs from published Gemini judgments"]}
    path = output / "run.json"
    def save():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(path)
    save()
    if args.dry_run:
        print(json.dumps({"status": "prepared", "api_calls": 0, "native_frames": frames}))
        return
    manifest = json.loads((ROOT / "reproduction/lingbot-weights.json").read_text())
    if manifest["revision"] != WEIGHT_REVISION:
        raise ValueError("wrong checkpoint inventory")
    raw = source / "checkpoints/lingbot-world-v2-14b-causal-fast"
    for item in manifest["files"]:
        if (raw / item["path"]).stat().st_size != item["bytes"]:
            raise ValueError("incomplete checkpoint")
    pack = runtime_pack(raw, source / "checkpoints/runtime-14b-causal-fast")
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from PIL import Image
    from einops import rearrange
    from worldsync.lingbot_vae import NativeVaeStream
    from worldsync.reference_player import decide
    sys.path.insert(0, str(source))
    from wan.configs import WAN_CONFIGS
    from wan.utils.cam_utils import compute_relative_poses, get_plucker_embeddings, get_Ks_transformed
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not stationary and not args.fixed_actions:
        from anthropic import Anthropic
        load_local_credentials()
        client = Anthropic(timeout=180, max_retries=2, default_headers={"Accept-Encoding": "identity"})
    controls = output / "controls"
    controls.mkdir()
    np.save(controls / "poses.npy", np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0))
    intrinsics = np.load(source / "examples/03/intrinsics.npy")[0]
    np.save(controls / "intrinsics.npy", np.repeat(intrinsics[None], frames, axis=0))
    started = time.monotonic()
    report["status"] = "running"
    save()
    handles = []
    try:
        upstream = configure_backend(report, save)
        pipeline = upstream.WanI2VCausal(WAN_CONFIGS["i2v-A14B"], str(pack),
            local_attn_size=18, sink_size=6, infer_mode="causal_fast", convert_model_dtype=True)
        stream = NativeVaeStream(pipeline.vae)
        camera = CameraPlan()
        current_chunk = -1
        latest = initial
        def before(model, positional, kwargs):
            nonlocal current_chunk
            index = int(kwargs["current_start"]) // (4 * int(kwargs["frame_seqlen"]))
            if index == current_chunk:
                return
            if index >= maximum or (not stationary and index >= 40):
                report["termination"] = "budget_exhausted"
                raise RolloutComplete
            if stationary:
                decision, metadata = {"operation": "KEEP", "action": "WAIT", "reason": "stationary observation"}, {}
            elif args.fixed_actions:
                if index >= len(args.fixed_actions):
                    report["termination"] = "fixed_diagnostic_actions_exhausted"
                    raise RolloutComplete
                decision, metadata = {"operation": "KEEP", "action": args.fixed_actions[index], "reason": "fixed diagnostic"}, {}
            else:
                decision, metadata = decide(client, task, initial, latest,
                    [{key: action[key] for key in ("operation", "action", "reason")} for action in report["actions"]])
                report["agent_interactions"] += 1
            report["actions"].append({**decision, **metadata, "elapsed_s": time.monotonic() - started})
            save()
            if decision["operation"] == "END":
                report["termination"] = "agent_end"
                raise RolloutComplete
            poses = torch.from_numpy(camera.advance(decision["action"])).float()
            relative = compute_relative_poses(poses, framewise=True)[-4:].to(pipeline.device)
            condition = kwargs["dit_cond_dict"]["c2ws_plucker_emb"][0]
            height, width = condition.shape[-2] * 8, condition.shape[-1] * 8
            ks = get_Ks_transformed(torch.from_numpy(intrinsics[None]).float(),
                height_org=480, width_org=832, height_resize=height, width_resize=width,
                height_final=height, width_final=width)[0].repeat(4, 1).to(pipeline.device)
            embedding = get_plucker_embeddings(relative, ks, height, width)
            # Match native rearrangement and channel strides as well as values.
            embedding = rearrange(embedding, "f (h a) (w b) c -> (f h w) (c a b)", a=8, b=8)[None]
            embedding = rearrange(embedding, "b (f h w) c -> b c f h w", f=4,
                h=condition.shape[-2], w=condition.shape[-1]).to(pipeline.param_dtype)
            if index == 0 and decision["action"] == "WAIT" and not torch.equal(embedding, condition):
                raise ValueError("neutral adapter controls differ from native camera embeddings")
            # This nested dict is shared by the native denoising and clean-cache
            # calls for this chunk. Only its external camera input is replaced.
            kwargs["dit_cond_dict"]["c2ws_plucker_emb"] = [embedding]
            current_chunk = index
        def after(model, positional, kwargs, result):
            nonlocal latest
            if float(kwargs["t"].flatten()[0]) != 0:
                return
            video = stream.decode(kwargs["x"][0])
            if not torch.isfinite(video).all():
                raise ValueError("non-finite decoded observation")
            pixels = ((video.permute(1, 2, 3, 0) + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
            index = len(report["chunks"])
            movie = output / f"chunk-{index:03d}.mp4"
            imageio.mimwrite(movie, pixels, fps=16, codec="libx264", quality=8, macro_block_size=1)
            if args.fixed_actions and maximum <= 2:
                np.save(output / f"pixels-{index:03d}.npy", pixels)
            latest = output / f"observation-{index:03d}.jpg"
            Image.fromarray(pixels[-1]).save(latest, quality=95)
            report["chunks"].append({"index": index, "path": movie.name, "frames": len(pixels),
                "pixel_sha256": hashlib.sha256(pixels).hexdigest(),
                "elapsed_s": time.monotonic() - started, "sha256": hashlib.sha256(movie.read_bytes()).hexdigest()})
            report["generated_frames"] = sum(chunk["frames"] for chunk in report["chunks"])
            report["generated_seconds"] = report["generated_frames"] / 16
            save()
            print(json.dumps({"event": "chunk_saved", **report["chunks"][-1]}), flush=True)
            if len(report["chunks"]) >= maximum:
                report["termination"] = "chunk_budget_exhausted"
                raise RolloutComplete
        handles = [pipeline.model.register_forward_pre_hook(before, with_kwargs=True),
                   pipeline.model.register_forward_hook(after, with_kwargs=True)]
        pipeline.generate(task["image_caption"], Image.open(initial).convert("RGB"),
            action_path=str(controls), chunk_size=4, max_area=480 * 832, frame_num=frames,
            shift=10, seed=42, offload_model=False)
    except RolloutComplete:
        report["status"] = "completed" if report["chunks"] else "empty_rollout"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        for handle in handles:
            handle.remove()
        report["elapsed_s"] = time.monotonic() - started
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        save()
        if report["chunks"]:
            listing = output / "chunks.txt"
            listing.write_text("".join(f"file '{chunk['path']}'\n" for chunk in report["chunks"]))
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                "-c", "copy", str(output / "rollout.mp4")], check=True)
    print(json.dumps({"status": report["status"], "chunks": len(report["chunks"])}))


if __name__ == "__main__":
    main()
