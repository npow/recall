"""Adaptive stock LingBot rollout through model I/O hooks; native KV policy retained."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

from worldsync.credentials import load_local_credentials
from worldsync.lingbot_controls import CameraPlan, fp32_camera_rays
from worldsync import lingbot_tuning
from worldsync.lingbot_diagnostics import summarize
from worldsync.lingbot_native import ROOT, SOURCE_REVISION, WEIGHT_REVISION, runtime_pack, configure_backend

PROTOCOL_ID = "lingbot-world2-stock-reconstructed-haiku-16fps-v6"
SCHEDULES = {"stock": {"timesteps_index": [0, 250, 500, 750], "shift": 10},
             "cmbench": {"timesteps_index": [0, 179, 358, 679], "shift": 5.0}}


class RolloutComplete(Exception):
    pass


def build_pipeline(upstream, configs, pack, knobs):
    """One-shot pipeline construction; the persistent runner supplies its own warm pipeline."""
    return upstream.WanI2VCausal(configs["i2v-A14B"], str(pack),
        local_attn_size=knobs["local_attn_size"], sink_size=knobs["sink_size"], infer_mode="causal_fast", convert_model_dtype=True)


def main(argv=None, runtime=None, player_decide=None, simulation_chunk_budget=None, controller_profile=None):
    """Run one rollout. `runtime` (worldsync/lingbot_runtime.py) may supply a warm pipeline; returns the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "external/PlayWorld/data")
    parser.add_argument("--split", choices=["gc", "if", "insight", "outsight"], required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--max-chunks", type=int, default=40)
    parser.add_argument("--fixed-actions", nargs="+", help="Transport/parity diagnostic; not adaptive benchmark evidence")
    parser.add_argument("--native-fixed-controls", action="store_true", help="Use native precomputed poses and audit online camera embeddings")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-prompt", choices=["caption", "prompt", "both"], default="caption",
        help="World-model text: caption = static image description (current baseline), prompt = task objective "
             "(what the released PlayWorld harness submits), both = caption then objective")
    parser.add_argument("--turn-clamp", choices=["off", "plan"], default="off",
        help="plan: limit commanded yaw to the range implied by the reference plan's Left/Right holds (50 deg/s pacing); "
             "the last turning chunk is shortened to land exactly on the limit")
    parser.add_argument("--director", choices=["off", "static", "perchunk", "unseen"], default="off",
        help="EXPERIMENTAL (worldsync/lingbot_director.py): an LLM writes the world-model text from the caption, task "
             "objective, reference plan, elapsed time and commanded camera state only (never rubric text)")
    parser.add_argument("--director-interval", type=int, default=4, help="chunks between director refreshes (perchunk/unseen)")
    parser.add_argument("--director-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--director-text-file", type=Path, help="use committed preset director texts instead of LLM calls")
    parser.add_argument("--nag", action="store_true",
        help="EXPERIMENTAL (worldsync/lingbot_nag.py): Normalized Attention Guidance with a negative prompt in cross-attention")
    parser.add_argument("--nag-scale", type=float, default=8.0)
    parser.add_argument("--nag-tau", type=float, default=2.5)
    parser.add_argument("--nag-alpha", type=float, default=0.25)
    parser.add_argument("--nag-negative", default=None, help="negative prompt (default: a short 'static, still, frozen' list)")
    parser.add_argument("--turn-degrees-per-second", type=float, default=60)
    parser.add_argument("--adaptive-retrieval-pilot", action="store_true",
                        help="Separate GC027 adaptive protocol:30deg/s, initial neutral chunk, optional frozen four-view memory")
    parser.add_argument("--stream-player-actions", action="store_true",
                        help="Overlap generation with the trailing streamed player explanation; same controller prompts and validation")
    parser.add_argument("--pose-retrieval", action="store_true", help="Experimental bounded early-view retrieval on native fixed paths")
    parser.add_argument("--retrieved-frames", type=int, choices=(4, 8), default=4)
    parser.add_argument("--retrieval-selection", choices=("nearest", "diverse"), default="nearest")
    parser.add_argument("--history-output-weight", type=float, choices=(.25, .5), default=None,
                        help="Declared22-frame8-view pilot: separately normalize recent/history attention")
    parser.add_argument("--archive-export", type=Path, help="Save this agent's retrieval archive and poses at close")
    parser.add_argument("--foreign-archive", action="append", default=[], metavar="NAME=PATH",
                        help="Also retrieve from another agent's exported archive (same seed/image/text/start pose)")
    parser.add_argument("--min-retrieved-frames", type=int, default=4,
                        help="Demo: apply retrieval when at least this many views are selected")
    parser.add_argument("--archive-until", type=int, default=22, help="Demo: archive latent frames [6, N)")
    parser.add_argument("--archive-stride", type=int, default=1, help="Demo: archive every k-th latent frame")
    parser.add_argument("--world-text-suffix", default="", help="Demo: text appended to the world-model caption")
    parser.add_argument("--retrieval-positions", choices=("virtual", "native"), default="virtual")
    parser.add_argument("--rephase-sink", action="store_true", help="Experimental pose-gated native sink key rephasing")
    parser.add_argument("--kv-retention", action="store_true", help="Diagnostic WorldKV-style relevance retrieval from an unbounded CPU KV archive (audited: noisy first-step query, 18/6 gate only)")
    parser.add_argument("--kv-retained-frames", type=int, default=4)
    parser.add_argument("--kv-rope", choices=["virtual", "native"], default="virtual")
    parser.add_argument("--decoprune", action="store_true", help="Diagnostic DeCoPrune KV pruning port (reproduction/decoprune-spec.md)")
    parser.add_argument("--decoprune-threshold", type=float, default=0.10)
    parser.add_argument("--decoprune-step", type=int, default=2, help="Zero-based denoising step whose x0 is scored")
    parser.add_argument("--decoprune-rope", choices=["all_bands", "off"], default="all_bands")
    parser.add_argument("--denoise-schedule", choices=list(SCHEDULES), default="stock",
        help="stock: shift 10, steps [0,250,500,750]; cmbench: DeCoPrune shift 5, steps [0,179,358,679]")
    parser.add_argument("--reanchor", action="store_true", help="Diagnostic start-view re-anchoring (frames 0-3 re-timed as recent)")
    parser.add_argument("--budget-memory", action="store_true", help="Bounded pose-indexed KV memory (sink + K retrieved + tail, evicting archive)")
    parser.add_argument("--budget-retrieved", type=int, default=10)
    parser.add_argument("--budget-archive", type=int, default=24)
    parser.add_argument("--budget-rope", choices=["virtual", "native"], default="virtual")
    parser.add_argument("--budget-store", default=None,
        help="Storage codec for archived memory frames (exact, fp8, int8, pool2, pool4, top25, top10, rand25, lr64, lr256, poolN+fp8); storage ablation, fixed native paths only")
    parser.add_argument("--budget-layers", default="all",
        help="Layers that archive and substitute memory (all, even, every4, first10, last10, first20, last20, mid20); storage ablation")
    parser.add_argument("--memory-v2", choices=["off", "disk", "recompute"], default="off",
        help="EXPERIMENTAL (worldsync/lingbot_memv2.py): content-verified retrieval over the whole episode; "
             "disk = exact K/V archived on local disk, recompute = clean latents kept and K/V recomputed when retrieved")
    parser.add_argument("--memv2-select", choices=["verified", "content", "pose"], default="verified")
    parser.add_argument("--memv2-retrieved", type=int, default=10)
    parser.add_argument("--memv2-rope", choices=["virtual", "native"], default="native")
    parser.add_argument("--memv2-descriptor", choices=["dino", "thumbnail"], default="dino")
    parser.add_argument("--memv2-stride", type=int, default=1)
    parser.add_argument("--memv2-tau-verify", type=float, default=0.75)
    parser.add_argument("--memv2-tau-direct", type=float, default=0.80)
    parser.add_argument("--memv2-tau-content", type=float, default=0.75)
    parser.add_argument("--memv2-store-dir", type=Path, help="directory for the temporary KV archive (default: WORLDSYNC_KV_DIR or the temp dir)")
    parser.add_argument("--offload-text-encoder", action="store_true", help="Release the completed GPU text encoder before denoising")
    parser.add_argument("--fp8-kv", action="store_true", help="Experimental fp8 (e4m3) self-attention KV cache; weights stay bf16 (native fixed paths)")
    parser.add_argument("--fp8-kv-quantize-current", action="store_true", help="Also attend to the fp8-rounded K/V of the chunk being generated (v1 behaviour; worse)")
    parser.add_argument("--fp8-kv-audit-layers", default="", help="Comma-separated layer indices that keep a bf16 shadow cache to measure attention error")
    parser.add_argument("--player-retries", type=int, default=0,
        help="Resample the player up to N times after an invalid response (same for every arm; each retry is logged)")
    parser.add_argument("--lora", type=Path, help="Memory LoRA adapter (scripts/train_lingbot_memory.py); non-stock weights")
    parser.add_argument("--memory-reader", type=Path, help="Experimental trained residual reader; standalone fixed diagnostics only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    lingbot_tuning.add_arguments(parser)
    args = parser.parse_args(argv)
    if simulation_chunk_budget is not None and (player_decide is None or type(simulation_chunk_budget) is not int or not 1 <= simulation_chunk_budget <= 60):
        parser.error("An explicit simulation budget requires a supplied player and1..60chunks")
    if player_decide is not None and args.stream_player_actions:
        parser.error("A supplied player callback cannot use the legacy streamed controller")
    if args.lora and not (args.controlled_ablation and args.native_fixed_controls and args.fixed_actions):
        parser.error("--lora changes the model; allowed only on fixed native-camera diagnostics with --controlled-ablation")
    if args.lora and not args.lora.is_file():
        parser.error("LoRA file not found")
    knobs = lingbot_tuning.resolve(args, parser)
    if args.memory_reader:
        if (runtime is not None or args.lora or not args.memory_reader.is_file()
                or not args.controlled_ablation or not args.native_fixed_controls or not args.fixed_actions
                or not args.pose_retrieval or args.retrieved_frames != 4
                or args.retrieval_positions != "virtual" or knobs["local_attn_size"] != 18
                or args.history_output_weight is not None):
            parser.error("memory reader requires standalone native fixed diagnostics, an existing checkpoint and virtual18/6 four-view retrieval")
    from worldsync import lingbot_adaptive_pilot
    lingbot_adaptive_pilot.validate(args, knobs, parser)
    if args.turn_clamp != "off" and not lingbot_tuning.hillclimb_unlocked():
        parser.error("turn clamping is locked until hillclimbing is unlocked")
    if args.text_prompt != "caption" and controller_profile is None and not lingbot_tuning.hillclimb_unlocked():
        parser.error("non-default world-model text is locked until hillclimbing is unlocked")
    if args.director != "off" and not lingbot_tuning.hillclimb_unlocked():
        parser.error("the text director is locked until hillclimbing is unlocked")
    if args.nag and not lingbot_tuning.hillclimb_unlocked():
        parser.error("NAG is locked until hillclimbing is unlocked")
    if args.director != "off" and args.fixed_actions and args.native_fixed_controls:
        parser.error("the text director is not supported with native fixed controls")
    if not 0 < args.turn_degrees_per_second <= 90:
        parser.error("turn rate must be in (0,90] degrees per second")
    if (args.turn_degrees_per_second != 60 and not args.native_fixed_controls
            and not args.adaptive_retrieval_pilot and not lingbot_tuning.hillclimb_unlocked()):
        parser.error("non-stock turn rates require a declared native/adaptive pilot or unlocked hillclimbing")
    if not 1 <= args.max_chunks <= 60:
        parser.error("chunk budget must be in [1,60]")
    if args.native_fixed_controls and (not args.fixed_actions or len(args.fixed_actions) != args.max_chunks):
        parser.error("native fixed controls require exactly max-chunks fixed actions")
    if (args.retrieved_frames != 4 or args.retrieval_selection != "nearest") and not args.pose_retrieval:
        parser.error("retrieval count/selection require --pose-retrieval")
    if args.retrieval_positions != "virtual" and not args.pose_retrieval:
        parser.error("retrieval positions require --pose-retrieval")
    if args.history_output_weight is not None and (not args.pose_retrieval or args.retrieved_frames != 8
            or knobs["local_attn_size"] != 22 or args.retrieval_selection != "nearest"
            or args.retrieval_positions != "virtual"):
        parser.error("gated attention pilot requires22 frames,8 nearest views,virtual positions and pose retrieval")
    if (args.budget_store is not None or args.budget_layers != "all") and not (
            args.budget_memory and args.native_fixed_controls and args.fixed_actions):
        parser.error("memory storage ablations require --budget-memory on fixed native-camera paths")
    diagnostics = [args.pose_retrieval, args.rephase_sink, args.kv_retention, args.decoprune, args.reanchor, args.budget_memory, args.fp8_kv,
                   args.memory_v2 != "off"]
    if sum(diagnostics) > 1:
        parser.error("memory ablations must be tested separately")
    adaptive_memory_ok = (args.budget_memory or args.memory_v2 != "off") and lingbot_tuning.hillclimb_unlocked()
    if ((any(diagnostics) or args.denoise_schedule != "stock") and not args.native_fixed_controls
            and not args.adaptive_retrieval_pilot and not adaptive_memory_ok):
        parser.error("retrieval requires native fixed controls, the declared adaptive pilot, or unlocked budget-memory")
    if args.rephase_sink and not lingbot_tuning.is_stock(knobs):
        parser.error("sink rephasing requires the stock 18-frame/6-sink budget")
    if args.pose_retrieval and (knobs["local_attn_size"] not in {18, 22}
            or knobs["sink_size"] != 6 or knobs["decision_interval"] != 1
            or (knobs["local_attn_size"] != 18 and not args.controlled_ablation)):
        parser.error("retrieval requires18/22 frames,6 sinks,decision interval1;22 requires controlled ablation")
    schedule = SCHEDULES[args.denoise_schedule]
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
    maximum = min(args.max_chunks, simulation_chunk_budget if simulation_chunk_budget is not None else (60 if stationary else 40))
    frames = maximum * 16 - 3
    report = {"status": "prepared", "model": "LingBot-World2 14B causal-fast stock", "protocol_id": lingbot_adaptive_pilot.PROTOCOL if args.adaptive_retrieval_pilot else PROTOCOL_ID,
        "source_revision": revision, "checkpoint_revision": WEIGHT_REVISION,
        "initial_image_sha256": hashlib.sha256(initial.read_bytes()).hexdigest(),
        "text_prompt_mode": args.text_prompt, "director_mode": args.director, "image_caption_sha256": hashlib.sha256(task["image_caption"].encode()).hexdigest(), "task_id": args.task_id,
        "split": args.split, "seed": args.seed, "turn_degrees_per_second": args.turn_degrees_per_second, "fps": 16, "max_chunks": maximum, "max_interactions": 40,
        "native_fixed_controls": args.native_fixed_controls, "camera_embedding_audit": [],
        "pose_retrieval": args.pose_retrieval, "sink_rephase": args.rephase_sink,
        "adaptive_retrieval_pilot": args.adaptive_retrieval_pilot,
        "player_temperature": "provider default",
        "player_streaming": args.stream_player_actions and not stationary and not args.fixed_actions,
        "streaming_player_sha256": (hashlib.sha256((Path(__file__).parent / "streaming_player.py").read_bytes()).hexdigest()
                                    if args.stream_player_actions else None),
        "initial_neutral_bootstrap": args.adaptive_retrieval_pilot,
        "kv_retention": args.kv_retention, "decoprune": args.decoprune, "reanchor": args.reanchor,
        "budget_memory": args.budget_memory, "budget_store": args.budget_store, "budget_layers": args.budget_layers, "fp8_kv": args.fp8_kv, "memory_v2": args.memory_v2, "denoise_schedule": args.denoise_schedule,
        "text_encoder_offloaded_after_encoding": args.offload_text_encoder,
        "stationary_wait_only": stationary, "fixed_actions": args.fixed_actions, "quantization": ("none (weights bf16); KV cache fp8 e4m3 [diagnostic, non-stock]" if args.fp8_kv else "none"), "cuda_allocator_configuration": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default"),
        "max_area": 480 * 832, "chunk_size": 4, "local_attn_size": knobs["local_attn_size"], "sink_size": knobs["sink_size"],
        "decision_interval": knobs["decision_interval"], "stock_knobs": lingbot_tuning.is_stock(knobs),
        "controlled_ablation": args.controlled_ablation,
        "sampler_timesteps_index": schedule["timesteps_index"], "sample_shift": schedule["shift"],
        "actions": [], "chunks": [], "published_score_reproduced": False,
        "agent_interactions": 0, "simulation_chunk_budget_override": simulation_chunk_budget,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "controls_sha256": hashlib.sha256((Path(__file__).parent / "lingbot_controls.py").read_bytes()).hexdigest(),
        "player_sha256": hashlib.sha256((Path(__file__).parent / "reference_player.py").read_bytes()).hexdigest(),
        "deviations": ["Hosted PlayWorld variant/settings not verified; official local 14B causal-fast checkpoint",
            f"Reconstructed Haiku reference-guided prompt and camera mapping: {args.turn_degrees_per_second:g} deg/s turns, unit-normalized translations",
            "Official 480p demo area and KV settings; single Thor GPU, BF16 and valid-length SDPA backend",
            "Native VAE sequential decoding exposed at chunk boundaries for observations; parity verified separately",
            "Claude diagnostic scoring differs from published Gemini judgments"]}
    if controller_profile is not None:
        if player_decide is None or args.decision_interval != 1:
            raise ValueError("Canonical controller requires callback on every native chunk")
        expected_text = 'prompt'
        if controller_profile.get('profile') == 'lingbot-insight-caption-ablation-v1':
            if (args.split != 'insight' or controller_profile.get('experimental_text_mode') != 'both'
                    or not args.controlled_ablation):
                raise ValueError('Caption ablation requires explicit in-sight experimental profile')
            expected_text = 'both'
        if (args.text_prompt != expected_text or not lingbot_tuning.is_stock(knobs)
                or any(diagnostics) or args.fixed_actions or args.native_fixed_controls
                or args.adaptive_retrieval_pilot or args.lora or args.memory_reader
                or args.turn_clamp != 'off' or args.denoise_schedule != 'stock'):
            raise ValueError('Canonical baseline requires stock model settings and its own camera schedule')
        if controller_profile['task_id'] != args.task_id or controller_profile['split'] != args.split:
            raise ValueError('Controller profile belongs to a different task')
        report.update(protocol_id=controller_profile['profile'], controller_profile=controller_profile,
                      player_sha256=controller_profile['source_sha256']['worldsync/released_reference_player.py'],
                      observation_encoding='lossless PNG before released JPEG stages',
                      initial_neutral_bootstrap=True)
        report['deviations'] = controller_profile['remaining_differences']
    path = output / "run.json"
    def save():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(path)
    save()
    if args.dry_run:
        print(json.dumps({"status": "prepared", "api_calls": 0, "native_frames": frames}))
        return report
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
    if controller_profile is None:
        report["deviations"][2] = f"Official 480p demo area; single {report['hardware']['gpu']}, BF16 and valid-length SDPA backend"
    from PIL import Image
    from einops import rearrange
    from worldsync.lingbot_vae import NativeVaeStream
    from worldsync.reference_player import decide as _decide, PlayerResponseError
    if player_decide is not None:
        _decide = player_decide
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from wan.configs import WAN_CONFIGS
    from wan.utils.cam_utils import compute_relative_poses, get_plucker_embeddings, get_Ks_transformed
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    client = None
    if (((not stationary or controller_profile is not None) and not args.fixed_actions)
            or (args.director != "off" and not args.director_text_file)):
        from anthropic import Anthropic
        load_local_credentials()
        client = Anthropic(timeout=180, max_retries=2, default_headers={"Accept-Encoding": "identity"})
    director = None
    director_context = None
    if args.director != "off":
        from worldsync.lingbot_director import Director, PresetDirector
        director = (PresetDirector(args.director_text_file, args.split, args.task_id, args.director) if args.director_text_file
                    else Director(client, task, args.director, model=args.director_model, interval=args.director_interval))
        report["deviations"].append(f"EXPERIMENTAL text director mode {args.director} (worldsync/lingbot_director.py): "
                                    "LLM-written world-model text; not the released PlayWorld text protocol")
        report["model"] = "LingBot-World2 14B causal-fast stock weights + text director (non-stock protocol)"
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
    torch.cuda.reset_peak_memory_stats()
    report["status"] = "running"
    save()
    handles = []
    memory = None
    nag = None
    pending_player = None
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
        if runtime is None:
            pipeline = build_pipeline(upstream, WAN_CONFIGS, pack, knobs)
        else:
            pipeline = runtime.pipeline_for(upstream, WAN_CONFIGS, pack, knobs, report)
        if args.lora:
            from worldsync import lingbot_lora
            report["lora"] = lingbot_lora.load_lora(pipeline.model, args.lora)
            report["model"] = "LingBot-World2 14B causal-fast + memory LoRA (non-stock)"
            save()
        if args.memory_reader:
            from worldsync import lingbot_memory_reader
            report["memory_reader"] = lingbot_memory_reader.load(pipeline.model, args.memory_reader)
            report["model"] = "LingBot-World2 14B causal-fast + learned memory reader (non-stock)"
            save()
        if args.nag:
            from worldsync.lingbot_nag import NAG, DEFAULT_NEGATIVE
            nag = NAG(pipeline, negative=args.nag_negative or DEFAULT_NEGATIVE, scale=args.nag_scale,
                      tau=args.nag_tau, alpha=args.nag_alpha).install()
            report["deviations"].append("EXPERIMENTAL Normalized Attention Guidance in cross-attention (worldsync/lingbot_nag.py)")
            report["model"] = "LingBot-World2 14B causal-fast stock weights + NAG (non-stock inference)"
            report["nag"] = nag.metadata()
        stream = NativeVaeStream(pipeline.vae)
        from worldsync.lingbot_controls import plan_yaw_limits
        yaw_limits = plan_yaw_limits(task) if args.turn_clamp == "plan" else None
        report["turn_clamp"] = {"mode": args.turn_clamp, "yaw_limits_degrees": yaw_limits}
        camera = CameraPlan(turn_degrees_per_second=args.turn_degrees_per_second, yaw_limits_degrees=yaw_limits)
        if args.pose_retrieval:
            from worldsync.lingbot_retrieval import PoseKVMemory
            memory = PoseKVMemory(pipeline, key_positions=args.retrieval_positions,
                                  retrieved_frames=args.retrieved_frames, selection=args.retrieval_selection,
                                  cache_frames=knobs["local_attn_size"], history_weight=args.history_output_weight,
                                  foreign_archives=dict(item.split("=", 1) for item in args.foreign_archive),
                                  export_path=args.archive_export, min_retrieved=args.min_retrieved_frames,
                                  archive_until=args.archive_until, archive_stride=args.archive_stride)
        elif args.rephase_sink:
            from worldsync.lingbot_sink_rephase import SinkRephaseMemory
            memory = SinkRephaseMemory(pipeline)
        elif args.kv_retention:
            from worldsync.lingbot_kvretention import RelevanceKVMemory
            memory = RelevanceKVMemory(pipeline, retained_frames=args.kv_retained_frames, rope=args.kv_rope)
        elif args.decoprune:
            from worldsync.lingbot_decoprune import DeCoPruneMemory
            memory = DeCoPruneMemory(pipeline, threshold=args.decoprune_threshold,
                generation_step_index=args.decoprune_step, rope=args.decoprune_rope)
        elif args.reanchor:
            from worldsync.lingbot_reanchor import ReanchorMemory
            memory = ReanchorMemory(pipeline)
        elif args.budget_memory:
            from worldsync.lingbot_budgetmem import BudgetPoseMemory, StoredBudgetMemory
            budget_arguments = dict(sink_size=knobs["sink_size"], local_attn_size=knobs["local_attn_size"],
                retrieved=args.budget_retrieved, archive_frames=args.budget_archive, rope=args.budget_rope)
            if args.budget_store is not None or args.budget_layers != "all":
                from worldsync import lingbot_memstore
                memory = StoredBudgetMemory(pipeline, lingbot_memstore.make_codec(args.budget_store or "exact"),
                    layers=None if args.budget_layers == "all" else lingbot_memstore.layer_set(args.budget_layers, len(pipeline.model.blocks)),
                    **budget_arguments)
            else:
                memory = BudgetPoseMemory(pipeline, **budget_arguments)
        elif args.fp8_kv:
            from worldsync.lingbot_fp8kv import Fp8KVCache
            audit = [int(item) for item in args.fp8_kv_audit_layers.split(",") if item.strip()]
            memory = Fp8KVCache(pipeline, audit_layers=audit, quantize_current=args.fp8_kv_quantize_current)
        elif args.memory_v2 != "off":
            from worldsync.lingbot_memv2 import MemoryV2
            from worldsync.lingbot_memv2_recompute import MemoryV2Recompute
            memory = (MemoryV2 if args.memory_v2 == "disk" else MemoryV2Recompute)(pipeline, sink_size=knobs["sink_size"], local_attn_size=knobs["local_attn_size"],
                retrieved=args.memv2_retrieved, rope=args.memv2_rope, select=args.memv2_select,
                descriptor=args.memv2_descriptor, stride=args.memv2_stride, tau_verify=args.memv2_tau_verify,
                tau_direct=args.memv2_tau_direct, tau_content=args.memv2_tau_content, store_root=args.memv2_store_dir)
            report["deviations"].append("EXPERIMENTAL memory v2 (worldsync/lingbot_memv2.py): exact per-layer K/V of every "
                "generated latent frame archived on local disk; retrieved frames replace part of the attended window")
            report["model"] = "LingBot-World2 14B causal-fast stock weights + memory v2 (non-stock attention window)"
        current_chunk = -1
        latest = initial
        def before(model, positional, kwargs):
            nonlocal current_chunk, pending_player, director_context
            if memory is not None and getattr(memory, "recomputing", False):
                return  # nested t=0 pass that recomputes retrieved K/V (memory v2 recompute)
            if memory:
                memory.clean = float(kwargs["t"].flatten()[0]) == 0
            index = int(kwargs["current_start"]) // (4 * int(kwargs["frame_seqlen"]))
            if index == current_chunk:
                if memory is not None and memory.clean and hasattr(memory, "store_clean"):
                    memory.store_clean(index, kwargs)
                if director_context is not None:
                    kwargs["context"] = director_context
                return
            if index >= maximum or (simulation_chunk_budget is None and not stationary and index >= 40):
                report["termination"] = "budget_exhausted"
                raise RolloutComplete
            if index == 0 and args.offload_text_encoder:
                pipeline.text_encoder.model.cpu()
                torch.cuda.empty_cache()
            if args.adaptive_retrieval_pilot and index == 0:
                decision, metadata = {"operation": "KEEP", "action": "WAIT", "reason": "declared neutral bootstrap"}, {"bootstrap": True}
            elif stationary and controller_profile is None:
                decision, metadata = {"operation": "KEEP", "action": "WAIT", "reason": "stationary observation"}, {}
            elif args.fixed_actions:
                if index >= len(args.fixed_actions):
                    report["termination"] = "fixed_diagnostic_actions_exhausted"
                    raise RolloutComplete
                decision, metadata = {"operation": "KEEP", "action": args.fixed_actions[index], "reason": "fixed diagnostic"}, {}
            elif (held := lingbot_tuning.held_decision(report["actions"], index, knobs["decision_interval"])):
                decision, metadata = held, {}
            else:
                history = [{key: action[key] for key in ("operation", "action", "reason")} for action in report["actions"]]
                commanded = (lingbot_adaptive_pilot.player_state(camera) if args.adaptive_retrieval_pilot
                             else camera.commanded_state())
                for attempt in range(args.player_retries + 1):
                    try:
                        if args.stream_player_actions:
                            from worldsync.streaming_player import StreamingDecision
                            pending_player = StreamingDecision(client, task, initial, latest, history, commanded_state=commanded)
                            decision, metadata = pending_player.action()
                        else:
                            decision, metadata = _decide(client, task, initial, latest, history, commanded_state=commanded)
                        break
                    except PlayerResponseError as error:
                        report.setdefault("player_retry_events", []).append(
                            {"chunk": index, "attempt": attempt, "error": str(error)})
                        if attempt == args.player_retries:
                            raise
                increment = metadata.get("agent_interaction_increment", 1)
                if type(increment) is not int or not 0 <= increment <= 1:
                    raise ValueError("Player must report zero or one interaction per callback")
                report["agent_interactions"] += increment
            report["actions"].append({**decision, **metadata, "pose_before": camera.commanded_state(),
                                      "elapsed_s": time.monotonic() - started})
            save()
            if decision["operation"] == "END":
                report["termination"] = "agent_end"
                raise RolloutComplete
            if controller_profile is not None:
                from worldsync.native_reference_timeline import apply_camera_steps
                poses = torch.from_numpy(apply_camera_steps(camera, metadata['native_camera_steps'])).float()
                # The native pipeline's input poses.npy remains neutral; the hook
                # replaces its rays below. Preserve the actual online pose prefix
                # separately so offline audits do not mistake that input for it.
                np.save(controls / 'online_latent_poses.npy', poses.numpy())
                report['camera_trajectory_artifacts'] = {
                    'initialization_only': 'controls/poses.npy',
                    'online_latent_poses': 'controls/online_latent_poses.npy',
                    'online_pose_samples': int(poses.shape[0]),
                    'scope': 'Commanded latent-step camera-to-world poses before relative conditioning; not measured visual motion or decoded-frame poses',
                }
            else:
                poses = torch.from_numpy(camera.advance(decision["action"])).float()
            if memory:
                memory.begin(index, camera.poses)
            report["actions"][-1]["pose_after"] = camera.commanded_state()
            save()
            if controller_profile is not None:
                from worldsync.camera_conditioning import fixed_scale_relative_poses
                relative = fixed_scale_relative_poses(compute_relative_poses, poses, metadata['translation_scale'])[-4:].to(pipeline.device)
            else:
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
            if director is not None and args.director != "static":
                # Per-chunk text: re-encode only when the director changed it, then rebuild the cross-attention
                # K/V cache exactly as upstream does on the first call of a generate().
                text, changed = director.for_chunk(index, decision["action"], camera.commanded_state(),
                                                   max(0.0, index - 0.1875) if index else 0.0)
                if changed:
                    # Encode at the chunk boundary; with --offload-text-encoder the encoder returns to the CPU
                    # straight away so the 80 GB cards keep their headroom during denoising.
                    pipeline.text_encoder.model.to(pipeline.device)
                    encoded = pipeline.text_encoder([text], pipeline.device)
                    director_context = [encoded[0].to(pipeline.device)]
                    if args.offload_text_encoder:
                        pipeline.text_encoder.model.cpu()
                        torch.cuda.empty_cache()
                    kwargs["cross_attn_first_call"] = True
                if director_context is not None:
                    kwargs["context"] = director_context
                report["director"] = director.metadata()
            if memory is not None and hasattr(memory, "prepare"):
                memory.prepare(kwargs)
            current_chunk = index
        def after(model, positional, kwargs, result):
            nonlocal latest, pending_player
            if memory is not None and getattr(memory, "recomputing", False):
                return
            if float(kwargs["t"].flatten()[0]) != 0:
                return
            decode_started = time.monotonic()
            if args.pose_retrieval or getattr(memory, "release_before_decode", False):
                memory.release_read_cache()
            video = stream.decode(kwargs["x"][0])
            if not torch.isfinite(video).all():
                raise ValueError("non-finite decoded observation")
            pixels = np.ascontiguousarray(((video.permute(1, 2, 3, 0) + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8))
            decoded_at = time.monotonic()
            index = len(report["chunks"])
            if memory is not None and hasattr(memory, "observe"):
                memory.observe(index, pixels)
            movie = output / f"chunk-{index:03d}.mp4"
            imageio.mimwrite(movie, pixels, fps=16, codec="libx264", quality=8, macro_block_size=1)
            encoded_at = time.monotonic()
            if args.fixed_actions and maximum <= 2:
                np.save(output / f"pixels-{index:03d}.npy", pixels)
            if controller_profile is not None:
                latest = output / f"observation-{index:03d}.png"
                Image.fromarray(pixels[-1]).save(latest)
            else:
                latest = output / f"observation-{index:03d}.jpg"
                Image.fromarray(pixels[-1]).save(latest, quality=95)
            if pending_player is not None:
                decision, metadata = pending_player.finish()
                report["actions"][-1].update({**decision, **metadata})
                pending_player.close()
                pending_player = None
            report["chunks"].append({"index": index, "path": movie.name, "frames": len(pixels),
                "pixel_sha256": hashlib.sha256(pixels).hexdigest(),
                "phase_seconds": {"vae_decode_and_download": decoded_at - decode_started,
                                  "mp4_encode": encoded_at - decoded_at},
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
        world_text = {"caption": task["image_caption"], "prompt": task["prompt"],
                      "both": f"{task['image_caption']} {task['prompt']}"}[args.text_prompt]
        if args.world_text_suffix:
            world_text = f"{world_text} {args.world_text_suffix}"
            report["world_text"] = world_text
            report["world_text_sha256"] = hashlib.sha256(world_text.encode()).hexdigest()
            report["deviations"].append("demo world-text suffix appended to caption")
        if director is not None:
            world_text = director.initial()  # replaces the static text; later chunks may refresh it
            report["director"] = director.metadata()
            report["world_text_sha256"] = hashlib.sha256(world_text.encode()).hexdigest()
            save()
        pipeline.generate(world_text, Image.open(initial).convert("RGB"),
            action_path=str(controls), chunk_size=4, max_area=480 * 832, frame_num=frames,
            shift=schedule["shift"], timesteps_index=schedule["timesteps_index"], seed=args.seed, offload_model=False)
    except RolloutComplete:
        report["status"] = "completed" if report["chunks"] else "empty_rollout"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        if hasattr(error, "metadata"):
            report["player_failure"] = error.metadata
        raise
    finally:
        if pending_player is not None:
            pending_player.close()
        if native_ray_builder is not None:
            upstream.get_plucker_embeddings = native_ray_builder
        if memory:
            memory.close()
        if nag is not None:
            report["nag"] = nag.metadata()
            nag.uninstall()
        for handle in handles:
            handle.remove()
        report["elapsed_s"] = time.monotonic() - started
        report["camera_diagnostics"] = summarize([a["action"] for a in report["actions"] if a.get("action")],
                                                  turn_degrees_per_second=args.turn_degrees_per_second,
                                                  yaw_limits_degrees=(report.get("turn_clamp") or {}).get("yaw_limits_degrees"))
        if controller_profile is not None:
            report['camera_diagnostics'] = camera.commanded_state()
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        save()
        if report["chunks"]:
            listing = output / "chunks.txt"
            listing.write_text("".join(f"file '{chunk['path']}'\n" for chunk in report["chunks"]))
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                "-c", "copy", str(output / "rollout.mp4")], check=True)
    print(json.dumps({"status": report["status"], "chunks": len(report["chunks"])}))
    return report


if __name__ == "__main__":
    main()
