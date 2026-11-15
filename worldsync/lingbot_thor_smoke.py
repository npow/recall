"""Isolated two-chunk Thor smoke, no archive, early one-shot T5 disposal."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time
import traceback


def meminfo():
    result = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value, *_ = line.split()
        result[name.rstrip(":")] = int(value) * 1024
    return result


def rss():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", choices=("smoke", "baseline", "file4"), default="smoke")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    import torch
    import worldsync.lingbot_rollout as rollout
    from worldsync.lingbot_local_memory import OneShotTextEncoder
    from worldsync.lingbot_vae import NativeVaeStream
    phase = {"name": "torch_imported"}
    stop = threading.Event()
    started = time.monotonic()
    observations = []
    lock = threading.Lock()
    summary = {"scope": "isolated local transport/memory experiment; no scored-quality claim",
               "status": "running", "torch": str(torch.__version__), "forced_flash_sdpa": True,
               "forced_flash_scope": "DiT self/cross attention only; native VAE backend defaults retained",
               "vae_attention_shapes": [],
               "model_weights_quantized": False, "native_kv_quantized": False,
               "training": False, "events": []}

    def sample(event=None):
        row = {"elapsed_s": time.monotonic() - started, "phase": phase["name"],
               "rss_bytes": rss(), "mem_available_bytes": meminfo()["MemAvailable"],
               "process_io": {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/self/io").read_text().splitlines()},
               "cuda_allocated_bytes": torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 0,
               "cuda_reserved_bytes": torch.cuda.memory_reserved() if torch.cuda.is_initialized() else 0}
        if event:
            row["event"] = event
        with lock:
            observations.append(row)
            with (output / "memory-samples.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        return row

    def mark(name):
        phase["name"] = name
        summary["events"].append(sample(name))

    def monitor():
        while not stop.wait(.25):
            sample()

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    original_backend = rollout.configure_backend
    original_decode = NativeVaeStream.decode
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    attention_bindings = []
    handles = []
    upstream = None
    constructor = None
    import worldsync.lingbot_retrieval as retrieval
    original_memory = retrieval.PoseKVMemory
    if args.episode == "file4":
        from worldsync.lingbot_file_archive import FileArchiveMemory
        class LocalFileMemory(FileArchiveMemory):
            def __init__(self, *values, **options):
                super().__init__(*values, archive_root=output / "archive-map", **options)
        retrieval.PoseKVMemory = LocalFileMemory
    summary["episode"] = args.episode

    def bounded_attention(query, key, value, *values, **options):
        if query.dtype == torch.float32 or query.shape[-1] > 256:
            import math
            scores = math.prod(query.shape[:-2]) * query.shape[-2] * key.shape[-2] * 4
            shape = {"query": list(query.shape), "key": list(key.shape), "fp32_score_matrix_bytes": scores}
            if shape not in summary["vae_attention_shapes"]:
                summary["vae_attention_shapes"].append(shape)
            if max(query.shape[-2], key.shape[-2]) > 8192 or scores > 1024 ** 3:
                raise ValueError("VAE attention exceeds the guarded local smoke-test matrix budget")
        return original_sdpa(query, key, value, *values, **options)

    def decode(stream, latent):
        mark("vae_decode_before")
        result = original_decode(stream, latent)
        mark("vae_decode_after")
        return result

    def configured(report, save):
        nonlocal upstream, constructor
        upstream = original_backend(report, save)
        from wan.modules import model_fast, model as base_model
        native_attention = model_fast.attention
        def forced_dit_attention(*values, **options):
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                return native_attention(*values, **options)
        for module, name in ((model_fast, "attention"), (model_fast, "flash_attention"), (base_model, "flash_attention")):
            attention_bindings.append((module, name, getattr(module, name)))
            setattr(module, name, forced_dit_attention)
        constructor = upstream.WanI2VCausal

        def create(*positional, **kwargs):
            mark("pipeline_loading")
            pipeline = constructor(*positional, **kwargs)
            summary["dit_parameter_bytes"] = sum(p.numel() * p.element_size() for p in pipeline.model.parameters())
            summary["dit_parameter_dtypes"] = sorted({str(p.dtype) for p in pipeline.model.parameters()})
            mark("pipeline_loaded")
            class Encoder(OneShotTextEncoder):
                def __call__(self, *values, **options):
                    mark("t5_encode_before")
                    context = super().__call__(*values, **options)
                    summary["t5_parameter_bytes_discarded"] = self.released_parameter_bytes
                    mark("t5_discarded_after_encode")
                    return context
            pipeline.text_encoder = Encoder(pipeline.text_encoder)
            original_kv = pipeline._initialize_self_kv_cache
            def initialize_kv(*values, **options):
                mark("native_kv_before")
                result = original_kv(*values, **options)
                mark("native_kv_after")
                return result
            pipeline._initialize_self_kv_cache = initialize_kv
            def before(model, positional, options):
                mark("clean_pass" if float(options["t"].flatten()[0]) == 0 else "denoising")
            handles.append(pipeline.model.register_forward_pre_hook(before, with_kwargs=True))
            report["local_memory_prototype"] = {"one_shot_t5_discard": True, "archive": "lossless-file4" if args.episode == "file4" else "none",
                                                 "forced_flash_sdpa": True, "not_scored": True}
            save()
            return pipeline
        upstream.WanI2VCausal = create
        return upstream

    try:
        mark("flash_preflight")
        with torch.inference_mode():
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                q = torch.randn(1, 40, 256, 128, dtype=torch.bfloat16, device="cuda")
                assert torch.isfinite(original_sdpa(q, q, q)).all()
                del q
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            mark("flash_preflight_passed")
            rollout.configure_backend = configured
            NativeVaeStream.decode = decode
            torch.nn.functional.scaled_dot_product_attention = bounded_attention
            controls = ["WAIT", "WAIT"] if args.episode == "smoke" else ["WAIT"] + ["RIGHT"] * 12
            sys.argv = ["lingbot_rollout", "--split", "gc", "--task-id", "GC007" if args.episode == "smoke" else "GC027", "--seed", "42",
                        "--max-chunks", str(len(controls)), "--native-fixed-controls", "--fixed-actions", *controls,
                        "--turn-degrees-per-second", "30",
                        "--offload-text-encoder", "--output", str(output / "rollout")]
            if args.episode == "file4":
                sys.argv += ["--pose-retrieval"]
            rollout.main()
        summary["status"] = "completed"
        summary["run"] = json.loads((output / "rollout/run.json").read_text())
    except BaseException:
        summary["status"] = "failed"
        summary["error"] = traceback.format_exc()
        raise
    finally:
        stop.set()
        watcher.join(timeout=2)
        for handle in handles:
            handle.remove()
        rollout.configure_backend = original_backend
        retrieval.PoseKVMemory = original_memory
        NativeVaeStream.decode = original_decode
        torch.nn.functional.scaled_dot_product_attention = original_sdpa
        for module, name, implementation in attention_bindings:
            setattr(module, name, implementation)
        if upstream is not None and constructor is not None:
            upstream.WanI2VCausal = constructor
        sample("finished")
        summary.update(elapsed_s=time.monotonic() - started,
                       min_mem_available_bytes=min(row["mem_available_bytes"] for row in observations),
                       peak_rss_bytes=max(row["rss_bytes"] for row in observations),
                       peak_sampled_cuda_allocated_bytes=max(row["cuda_allocated_bytes"] for row in observations),
                       peak_sampled_cuda_reserved_bytes=max(row["cuda_reserved_bytes"] for row in observations))
        (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
