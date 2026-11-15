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
from worldsync.lingbot_controls import CameraPlan, fp32_camera_rays
from worldsync import lingbot_tuning
from worldsync.lingbot_diagnostics import summarize
from worldsync.lingbot_native import ROOT, SOURCE_REVISION, WEIGHT_REVISION, runtime_pack, configure_backend

PROTOCOL_ID = "lingbot-world2-stock-reconstructed-haiku-16fps-v5"


class RolloutComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "external/PlayWorld/data")
    parser.add_argument("--split", choices=["gc", "if", "insight", "outsight"], required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--max-chunks", type=int, default=40)
    parser.add_argument("--fixed-actions", nargs="+", help="Transport/parity diagnostic; not adaptive benchmark evidence")
    parser.add_argument("--native-fixed-controls", action="store_true", help="Use native precomputed poses and audit online camera embeddings")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--turn-degrees-per-second", type=float, default=60)
    parser.add_argument("--pose-retrieval", action="store_true", help="Experimental bounded early-view retrieval on native fixed paths")
    parser.add_argument("--retrieval-positions", choices=("virtual", "native"), default="virtual")
    parser.add_argument("--rephase-sink", action="store_true", help="Experimental pose-gated native sink key rephasing")
    parser.add_argument("--offload-text-encoder", action="store_true", help="Release the completed GPU text encoder before denoising")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    lingbot_tuning.add_arguments(parser)
    args = parser.parse_args()
    knobs = lingbot_tuning.resolve(args, parser)
    if not 0 < args.turn_degrees_per_second <= 90:
        parser.error("turn rate must be in (0,90] degrees per second")
    if args.turn_degrees_per_second != 60 and not args.native_fixed_controls:
        parser.error("non-stock turn rates are confined to declared native fixed-path tests")
    if not 1 <= args.max_chunks <= 60:
        parser.error("chunk budget must be in [1,60]")
    if args.native_fixed_controls and (not args.fixed_actions or len(args.fixed_actions) != args.max_chunks):
        parser.error("native fixed controls require exactly max-chunks fixed actions")
    if args.retrieval_positions != "virtual" and not args.pose_retrieval:
        parser.error("retrieval positions require --pose-retrieval")
    if args.pose_retrieval and args.rephase_sink:
        parser.error("memory ablations must be tested separately")
    if (args.pose_retrieval or args.rephase_sink) and not args.native_fixed_controls:
        parser.error("retrieval calibration requires native fixed controls")
    if (args.pose_retrieval or args.rephase_sink) and not lingbot_tuning.is_stock(knobs):
        parser.error("bounded retrieval currently requires the stock 18-frame/6-sink budget")
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
        "split": args.split, "seed": args.seed, "turn_degrees_per_second": args.turn_degrees_per_second, "fps": 16, "max_chunks": maximum, "max_interactions": 40,
        "native_fixed_controls": args.native_fixed_controls, "camera_embedding_audit": [],
        "pose_retrieval": args.pose_retrieval, "sink_rephase": args.rephase_sink,
        "text_encoder_offloaded_after_encoding": args.offload_text_encoder,
        "stationary_wait_only": stationary, "fixed_actions": args.fixed_actions, "quantization": "none",
        "max_area": 480 * 832, "chunk_size": 4, "local_attn_size": knobs["local_attn_size"], "sink_size": knobs["sink_size"],
        "decision_interval": knobs["decision_interval"], "stock_knobs": lingbot_tuning.is_stock(knobs),
        "controlled_ablation": args.controlled_ablation,
        "sampler_timesteps_index": [0, 250, 500, 750], "sample_shift": 10,
        "actions": [], "chunks": [], "published_score_reproduced": False,
        "agent_interactions": 0,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "controls_sha256": hashlib.sha256((Path(__file__).parent / "lingbot_controls.py").read_bytes()).hexdigest(),
        "player_sha256": hashlib.sha256((Path(__file__).parent / "reference_player.py").read_bytes()).hexdigest(),
        "deviations": ["Hosted PlayWorld variant/settings not verified; official local 14B causal-fast checkpoint",
            f"Reconstructed Haiku reference-guided prompt and camera mapping: {args.turn_degrees_per_second:g} deg/s turns, unit-normalized translations",
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
    report["hardware"] = {"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "cuda": torch.version.cuda}
    report["deviations"][2] = f"Official 480p demo area; single {report['hardware']['gpu']}, BF16 and valid-length SDPA backend"
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
    if args.native_fixed_controls:
        from worldsync.lingbot_paths import native_frame_poses
        native_poses, expected_final = native_frame_poses(args.fixed_actions, turn_degrees_per_second=args.turn_degrees_per_second)
        np.save(controls / "poses.npy", native_poses)
        report["expected_final_commanded_state"] = expected_final
    intrinsics = np.load(source / "examples/03/intrinsics.npy")[0]
    np.save(controls / "intrinsics.npy", np.repeat(intrinsics[None], frames, axis=0))
    started = time.monotonic()
    report["status"] = "running"
    save()
    handles = []
    memory = None
    native_ray_builder = None
    native_rays = {}
    try:
        upstream = configure_backend(report, save)
        if args.native_fixed_controls:
            # Observe the exact native FP32 ray tensor before its BF16 cast.
            # Return the same tensor object and keep native generation untouched.
            native_ray_builder = upstream.get_plucker_embeddings
            def capture_native_rays(*positional, **kwargs):
                rays = native_ray_builder(*positional, **kwargs)
                native_rays["fp32"] = rays.detach().cpu()
                return rays
            upstream.get_plucker_embeddings = capture_native_rays
        pipeline = upstream.WanI2VCausal(WAN_CONFIGS["i2v-A14B"], str(pack),
            local_attn_size=knobs["local_attn_size"], sink_size=knobs["sink_size"], infer_mode="causal_fast", convert_model_dtype=True)
        stream = NativeVaeStream(pipeline.vae)
        camera = CameraPlan(turn_degrees_per_second=args.turn_degrees_per_second)
        if args.pose_retrieval:
            from worldsync.lingbot_retrieval import PoseKVMemory
            memory = PoseKVMemory(pipeline, key_positions=args.retrieval_positions)
        elif args.rephase_sink:
            from worldsync.lingbot_sink_rephase import SinkRephaseMemory
            memory = SinkRephaseMemory(pipeline)
        current_chunk = -1
        latest = initial
        def before(model, positional, kwargs):
            nonlocal current_chunk
            if memory:
                memory.clean = float(kwargs["t"].flatten()[0]) == 0
            index = int(kwargs["current_start"]) // (4 * int(kwargs["frame_seqlen"]))
            if index == current_chunk:
                return
            if index >= maximum or (not stationary and index >= 40):
                report["termination"] = "budget_exhausted"
                raise RolloutComplete
            if index == 0 and args.offload_text_encoder:
                pipeline.text_encoder.model.cpu()
                torch.cuda.empty_cache()
            if stationary:
                decision, metadata = {"operation": "KEEP", "action": "WAIT", "reason": "stationary observation"}, {}
            elif args.fixed_actions:
                if index >= len(args.fixed_actions):
                    report["termination"] = "fixed_diagnostic_actions_exhausted"
                    raise RolloutComplete
                decision, metadata = {"operation": "KEEP", "action": args.fixed_actions[index], "reason": "fixed diagnostic"}, {}
            elif (held := lingbot_tuning.held_decision(report["actions"], index, knobs["decision_interval"])):
                decision, metadata = held, {}
            else:
                decision, metadata = decide(client, task, initial, latest,
                    [{key: action[key] for key in ("operation", "action", "reason")} for action in report["actions"]],
                    commanded_state=camera.commanded_state())
                report["agent_interactions"] += 1
            report["actions"].append({**decision, **metadata, "pose_before": camera.commanded_state(),
                                      "elapsed_s": time.monotonic() - started})
            save()
            if decision["operation"] == "END":
                report["termination"] = "agent_end"
                raise RolloutComplete
            poses = torch.from_numpy(camera.advance(decision["action"])).float()
            if memory:
                memory.begin(index, camera.poses)
            report["actions"][-1]["pose_after"] = camera.commanded_state()
            save()
            relative = compute_relative_poses(poses, framewise=True)[-4:].to(pipeline.device)
            condition = kwargs["dit_cond_dict"]["c2ws_plucker_emb"][0]
            height, width = condition.shape[-2] * 8, condition.shape[-1] * 8
            ks = get_Ks_transformed(torch.from_numpy(intrinsics[None]).float(),
                height_org=480, width_org=832, height_resize=height, width_resize=width,
                height_final=height, width_final=width)[0].repeat(4, 1).to(pipeline.device)
            # Native camera rays are built before the denoiser's BF16 autocast
            # scope. Hooks run inside it; preserve native FP32 ray arithmetic.
            rays_fp32 = fp32_camera_rays(get_plucker_embeddings, relative, ks, height, width)
            embedding = rays_fp32
            # Match native rearrangement and channel strides as well as values.
            embedding = rearrange(embedding, "f (h a) (w b) c -> (f h w) (c a b)", a=8, b=8)[None]
            embedding = rearrange(embedding, "b (f h w) c -> b c f h w", f=4,
                h=condition.shape[-2], w=condition.shape[-1]).to(pipeline.param_dtype)
            if index == 0 and decision["action"] == "WAIT" and not torch.equal(embedding, condition):
                raise ValueError("neutral adapter controls differ from native camera embeddings")
            if args.native_fixed_controls:
                from worldsync.lingbot_camera_audit import audit_rays
                native_fp32 = native_rays["fp32"][index * 4:(index + 1) * 4].to(pipeline.device)
                native_channels = rearrange(native_fp32, "f (h a) (w b) c -> (f h w) (c a b)", a=8, b=8)[None]
                native_channels = rearrange(native_channels, "b (f h w) c -> b c f h w", f=4,
                    h=condition.shape[-2], w=condition.shape[-1])
                online_channels = rearrange(rays_fp32, "f (h a) (w b) c -> (f h w) (c a b)", a=8, b=8)[None]
                online_channels = rearrange(online_channels, "b (f h w) c -> b c f h w", f=4,
                    h=condition.shape[-2], w=condition.shape[-1])
                audit = {"chunk": index, **audit_rays(online_channels, native_channels, condition)}
                report["camera_embedding_audit"].append(audit)
                if not audit["within_tolerance"]:
                    raise ValueError(f"online camera conditioning differs from native at chunk {index}: {audit}")
            # This nested dict is shared by the native denoising and clean-cache
            # calls for this chunk. Only its external camera input is replaced.
            if not args.native_fixed_controls:
                kwargs["dit_cond_dict"]["c2ws_plucker_emb"] = [embedding]
            current_chunk = index
        def after(model, positional, kwargs, result):
            nonlocal latest
            if float(kwargs["t"].flatten()[0]) != 0:
                return
            video = stream.decode(kwargs["x"][0])
            if not torch.isfinite(video).all():
                raise ValueError("non-finite decoded observation")
            pixels = np.ascontiguousarray(((video.permute(1, 2, 3, 0) + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8))
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
            if memory:
                report["memory"] = memory.metadata()
            save()
            print(json.dumps({"event": "chunk_saved", **report["chunks"][-1]}), flush=True)
            if len(report["chunks"]) >= maximum:
                report["termination"] = "chunk_budget_exhausted"
                raise RolloutComplete
        handles = [pipeline.model.register_forward_pre_hook(before, with_kwargs=True),
                   pipeline.model.register_forward_hook(after, with_kwargs=True)]
        pipeline.generate(task["image_caption"], Image.open(initial).convert("RGB"),
            action_path=str(controls), chunk_size=4, max_area=480 * 832, frame_num=frames,
            shift=10, seed=args.seed, offload_model=False)
    except RolloutComplete:
        report["status"] = "completed" if report["chunks"] else "empty_rollout"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        if hasattr(error, "metadata"):
            report["player_failure"] = error.metadata
        raise
    finally:
        if native_ray_builder is not None:
            upstream.get_plucker_embeddings = native_ray_builder
        if memory:
            memory.close()
        for handle in handles:
            handle.remove()
        report["elapsed_s"] = time.monotonic() - started
        report["camera_diagnostics"] = summarize([a["action"] for a in report["actions"] if a.get("action")], turn_degrees_per_second=args.turn_degrees_per_second)
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
