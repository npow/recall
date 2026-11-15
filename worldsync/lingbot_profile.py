"""Profile warmed stock inference without changing sampler, precision, or cache."""
import json
import os
from pathlib import Path
import sys
import time


def main():
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    import worldsync.lingbot_rollout as rollout
    from worldsync.lingbot_vae import NativeVaeStream

    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # The rollout requires an empty directory until its own initialization.
    original_backend = rollout.configure_backend
    profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       record_shapes=True, with_stack=False, profile_memory=False)
    active = False
    finished = False
    ranges = []
    handles = []
    metadata = {"scope": "diagnostic profiling; not scored baseline", "start_chunk": 6, "stop_chunk": 8}

    def stop():
        nonlocal active, finished
        if not active:
            return
        profiler.stop()
        active, finished = False, True
        profiler.export_chrome_trace(str(output / "trace.json"))
        events = profiler.key_averages(group_by_input_shape=True)
        (output / "operators.txt").write_text(events.table(sort_by="self_cuda_time_total", row_limit=50))
        rows = []
        for event in events:
            rows.append({"name": event.key, "calls": event.count, "input_shapes": event.input_shapes,
                         "device_type": str(event.device_type),
                         "self_cpu_us": event.self_cpu_time_total, "cpu_total_us": event.cpu_time_total,
                         "self_device_us": event.self_device_time_total,
                         "device_total_us": event.device_time_total})
        metadata["operators"] = sorted(rows, key=lambda row: row["self_device_us"], reverse=True)
        metadata["profile_completed"] = True

    def configured(report, save):
        upstream = original_backend(report, save)
        metadata["host_cache_indices"] = os.environ.get("WORLDSYNC_HOST_CACHE_INDICES") == "1"
        if metadata["host_cache_indices"]:
            from worldsync.lingbot_fast_metadata import install
            install(upstream)
        from wan.modules import model_fast, model as base_model
        original_attention = model_fast.attention

        def attention(*args, **kwargs):
            with record_function("worldsync/attention"):
                return original_attention(*args, **kwargs)

        model_fast.attention = model_fast.flash_attention = base_model.flash_attention = attention
        original_constructor = upstream.WanI2VCausal

        def constructor(*args, **kwargs):
            started = time.monotonic()
            pipeline = original_constructor(*args, **kwargs)
            metadata["pipeline_load_wall_s"] = time.monotonic() - started

            def before(model, positional, arguments):
                nonlocal active
                index = int(arguments["current_start"]) // (4 * int(arguments["frame_seqlen"]))
                if index >= 8 and active:
                    stop()
                if index == 6 and not active and not finished:
                    profiler.start()
                    active = True
                if active:
                    label = "clean_cache" if float(arguments["t"].flatten()[0]) == 0 else "denoise"
                    context = record_function(f"worldsync/dit_{label}")
                    context.__enter__()
                    ranges.append(context)

            def after(model, positional, arguments, result):
                if ranges:
                    ranges.pop().__exit__(None, None, None)

            handles.extend([pipeline.model.register_forward_pre_hook(before, with_kwargs=True),
                            pipeline.model.register_forward_hook(after, with_kwargs=True)])
            return pipeline

        upstream.WanI2VCausal = constructor
        return upstream

    rollout.configure_backend = configured
    original_decode = NativeVaeStream.decode

    def decode(self, latent):
        with record_function("worldsync/vae_decode"):
            return original_decode(self, latent)

    NativeVaeStream.decode = decode
    started = time.monotonic()
    try:
        rollout.main()
    finally:
        while ranges:
            ranges.pop().__exit__(None, None, None)
        stop()
        for handle in handles:
            handle.remove()
        metadata.update(wall_s=time.monotonic() - started, torch=str(torch.__version__),
                        gpu=torch.cuda.get_device_name(), profile_completed=finished)
        (output / "profile.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
