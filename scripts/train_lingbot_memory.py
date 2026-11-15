"""LoRA fine-tuning of LingBot-World2 14B causal-fast for memory-keyframe revisits.

Runs on one GPU (see scripts/modal_lingbot_train.py). Data: --manifest (real clips) or
--synthetic-tasks (pan-out-and-back clips from PlayWorld GC images; harness validation only).
Reports: stock-parity check, loss before/after per memory mode, throughput, peak memory,
and a LoRA save/load round trip. Not a PlayWorld score.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_pipeline(report):
    from worldsync.lingbot_native import ROOT as NATIVE_ROOT, runtime_pack, configure_backend
    source = NATIVE_ROOT / "external/LingBot-World2"
    sys.path.insert(0, str(source))
    from wan.configs import WAN_CONFIGS
    raw = source / "checkpoints/lingbot-world-v2-14b-causal-fast"
    pack = runtime_pack(raw, source / "checkpoints/runtime-14b-causal-fast")
    upstream = configure_backend(report, lambda: None)
    return upstream.WanI2VCausal(WAN_CONFIGS["i2v-A14B"], str(pack), local_attn_size=18, sink_size=6,
        infer_mode="causal_fast", convert_model_dtype=True)


def encode_prompt(pipeline, prompt):
    import torch
    pipeline.text_encoder.model.to(pipeline.device)
    with torch.no_grad():
        embedding = pipeline.text_encoder([prompt], pipeline.device)[0].detach()
    pipeline.text_encoder.model.cpu()
    torch.cuda.empty_cache()
    return embedding


def parity(pipeline, clip):
    """Our teacher-forced forward vs the stock KV-cache forward: chunk 1 given clean chunk 0."""
    import torch
    from worldsync.lingbot_memory_train import Group, context_kv, schedule, target_flow, text_context
    model = pipeline.model
    timesteps = schedule(pipeline)
    t = timesteps[2]
    frames0, frames1 = list(range(0, 4)), list(range(4, 8))
    generator = torch.Generator(device=clip.latent.device).manual_seed(7)
    x0 = clip.latent[:, frames1].float()
    noisy = pipeline.scheduler.add_noise(x0, torch.randn(x0.shape, generator=generator, device=x0.device), t)
    frame_tokens = (clip.latent.shape[2] // 2) * (clip.latent.shape[3] // 2)
    kv_size = frame_tokens * 18
    heads, head_dim = model.config.num_heads, model.config.dim // model.config.num_heads
    self_cache = pipeline._initialize_self_kv_cache(model.config.num_layers, [1, kv_size, heads, head_dim], torch.bfloat16, pipeline.device)
    cross_cache = pipeline._initialize_crossattn_cache(model.config.num_layers, [1, 512, heads, head_dim], torch.bfloat16, pipeline.device)
    with torch.amp.autocast("cuda", dtype=pipeline.param_dtype), torch.no_grad():
        def stock(latent, y, plucker, timestep, start_chunk, first):
            return model(x=[latent], t=torch.as_tensor([float(timestep)], device=pipeline.device), context=[clip.text],
                seq_len=4 * frame_tokens, y=[y], dit_cond_dict={"c2ws_plucker_emb": plucker.chunk(1, dim=0)},
                kv_cache=self_cache, crossattn_cache=cross_cache, current_start=start_chunk * 4 * frame_tokens,
                max_attention_size=kv_size, frame_seqlen=frame_tokens, cross_attn_first_call=first)[0]
        l0, y0, p0 = clip.latent[:, frames0], clip.y[:, frames0], clip.plucker[:, :, frames0]
        stock(l0.float(), y0, p0, 0.0, 0, True)
        reference = stock(noisy, clip.y[:, frames1], clip.plucker[:, :, frames1], t, 1, False)
        text = text_context(model, clip.text)
        context = [Group(model, l0, y0, p0, frames0, 0.0)]
        mine_target = Group(model, noisy, clip.y[:, frames1], clip.plucker[:, :, frames1], frames1, t)
        mine = target_flow(model, mine_target, context_kv(model, context, text), text, grad_checkpoint=False)
    error = (mine - reference.float()).abs()
    return {"max_abs_error": float(error.max()), "mean_abs_error": float(error.mean()),
            "reference_mean_abs": float(reference.float().abs().mean()),
            "relative_mean_error": float(error.mean() / reference.float().abs().mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic-tasks", nargs="*", default=[], help="GC task ids used as synthetic training clips")
    parser.add_argument("--heldout-tasks", nargs="*", default=[], help="GC task ids evaluated but never trained on")
    parser.add_argument("--manifest", type=Path, help="jsonl of real clips (see worldsync.lingbot_memory_data)")
    parser.add_argument("--data-root", type=Path, default=Path("/data"))
    parser.add_argument("--content-ok-only", action="store_true", help="Train only on manifest pairs flagged content_ok")
    parser.add_argument("--heldout-count", type=int, default=0, help="Hold out this many manifest pairs (whole source videos)")
    parser.add_argument("--max-train", type=int, default=0, help="Cap on manifest training clips")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--loss", choices=["velocity", "x0"], default="velocity")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    from worldsync import lingbot_lora, lingbot_memory_data as data
    from worldsync.lingbot_memory_train import evaluate, memory_loss
    report = {"status": "running", "args": {k: str(v) for k, v in vars(args).items()}, "scope": "harness validation; not a PlayWorld score"}
    path = args.output / "train.json"

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n")

    started = time.monotonic()
    pipeline = build_pipeline(report)
    model = pipeline.model
    model.eval()
    report["load_s"] = time.monotonic() - started
    report["gpu"] = torch.cuda.get_device_name(0)
    save()
    intrinsics = np.load(ROOT / "external/LingBot-World2/examples/03/intrinsics.npy")[0]
    tasks = {r["task_id"]: r for r in json.loads((ROOT / "external/PlayWorld/data/gc/data.json").read_text())}

    def synthetic(task_id):
        task = tasks[task_id]
        frames, poses, _ = data.synthetic_pan_clip(ROOT / "external/PlayWorld/data" / task["image_path"], intrinsics)
        return data.encode_clip(pipeline, frames, poses, intrinsics, encode_prompt(pipeline, task["image_caption"]),
            revisit_chunk=24, memory_start=8, wrong_start=4, name=f"synthetic-pan-{task_id}")

    train_clips = [synthetic(t) for t in args.synthetic_tasks]
    if args.manifest:
        entries = data.read_manifest(args.manifest, args.content_ok_only)
        train_entries, held_entries = data.split_manifest(entries, args.heldout_count, args.max_train or None)
        fallback = encode_prompt(pipeline, "A first-person walk through a real place, photorealistic, stable scene.")
        load = lambda e: data.clip_from_entry(pipeline, e, encode_prompt(pipeline, e["prompt"]) if "prompt" in e else fallback,
            intrinsics, args.data_root)
        train_clips += [load(e) for e in train_entries]
        manifest_heldout = [load(e) for e in held_entries]
    else:
        manifest_heldout = []
    heldout = [synthetic(t) for t in args.heldout_tasks] + manifest_heldout
    if not train_clips:
        parser.error("no training clips")
    report["clips"] = {"train": [c.name for c in train_clips], "heldout": [c.name for c in heldout]}
    report["encode_s"] = time.monotonic() - started - report["load_s"]
    save()
    torch.cuda.reset_peak_memory_stats()
    if args.parity:
        report["parity_vs_stock"] = parity(pipeline, train_clips[0])
        print(json.dumps(report["parity_vs_stock"]), flush=True)
        save()

    def snapshot(label):
        report[label] = {c.name: evaluate(pipeline, c, loss_space=args.loss) for c in heldout + train_clips[:1]}
        print(json.dumps({label: report[label]}), flush=True)
        save()

    snapshot("eval_before")  # adapter not attached yet: exactly the stock model
    trainable = lingbot_lora.apply_lora(model, args.rank, args.alpha)
    report["trainable_parameters"] = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    rng = random.Random(args.seed)
    generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)
    curve, step_times = [], []
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.steps + 1):
        clip = train_clips[(step - 1) % len(train_clips)]
        index = rng.randrange(4)
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss = memory_loss(pipeline, clip, "correct", index, generator, args.loss)
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        optimizer.step()
        torch.cuda.synchronize()
        step_times.append(time.monotonic() - tick)
        curve.append({"step": step, "clip": clip.name, "schedule_index": index, "loss": float(loss), "grad_norm": norm})
        if step % args.log_every == 0 or step == 1:
            print(json.dumps(curve[-1]), flush=True)
            report["curve"] = curve
            save()
    report["curve"] = curve
    timed = step_times[2:] or step_times
    report["train"] = {"steps": args.steps, "mean_step_s": sum(timed) / len(timed), "steps_per_s": len(timed) / sum(timed),
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9}
    snapshot("eval_after")
    lora_path = args.output / "lora.pt"
    digest = lingbot_lora.save_lora(model, lora_path, extra={"steps": args.steps, "clips": report["clips"], "loss": args.loss})
    probe = heldout[0] if heldout else train_clips[0]
    measure = lambda: evaluate(pipeline, probe, modes=("correct",), seeds=(0,), loss_space=args.loss)["correct"]["mean"]
    trained = measure()
    for p in trainable:
        p.data.zero_()
    zeroed = measure()
    meta = lingbot_lora.load_lora(model, lora_path)
    reloaded = measure()
    report["lora_roundtrip"] = {"sha256": digest, "loaded": meta["sha256"], "loss_trained": trained, "loss_zeroed": zeroed,
        "loss_reloaded": reloaded, "reload_matches": abs(trained - reloaded) < 1e-6 * max(1.0, abs(trained))}
    report["status"] = "completed"
    report["elapsed_s"] = time.monotonic() - started
    save()
    print(json.dumps({k: report[k] for k in ("status", "train", "lora_roundtrip")}, indent=2))


if __name__ == "__main__":
    main()
