"""Measure fused attention kernels; invoked in two isolated Torch environments."""
import argparse
import json
from pathlib import Path


def benchmark(include_fa4):
    import importlib.metadata
    import statistics
    import time
    import traceback

    import torch
    from matrix_attention import sdpa_attention

    report = {"scope": "synthetic kernel microbenchmark at recorded model shapes; not rollout speed or quality",
              "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
              "cuda": torch.version.cuda, "dtype": "bfloat16", "seed": 731,
              "training": False, "quantization": False,
              "optional_torch_c_dlpack_addon": include_fa4,
              "cxx11_abi": torch.compiled_with_cxx11_abi(),
              "versions": ({name: importlib.metadata.version(name) for name in
                           ("flash-attn-4", "nvidia-cutlass-dsl", "quack-kernels", "apache-tvm-ffi", "torch-c-dlpack-ext")} if include_fa4 else {}),
              "cases": [], "status": "running"}
    try:
        if include_fa4:
            from flash_attn.cute import flash_attn_func

        def fa4(q, k, v):
            result = flash_attn_func(q, k, v, causal=False)
            return result[0] if isinstance(result, tuple) else result

        torch.manual_seed(report["seed"])
        backends = {"sdpa": sdpa_attention}
        if include_fa4:
            backends["fa4"] = fa4
        with torch.inference_mode():
            for name, length in (("self4", 6032), ("self18", 27144),
                                 ("self22", 33176), ("self34", 51272), ("cross512", 512)):
                q = torch.randn(1, 6032, 40, 128, device="cuda", dtype=torch.bfloat16)
                k = torch.randn(1, length, 40, 128, device="cuda", dtype=torch.bfloat16)
                v = torch.randn_like(k)
                case = {"name": name, "q_shape": list(q.shape), "kv_shape": list(k.shape),
                        "measurements": {}, "accuracy": {}}
                outputs = {}
                for backend_name, function in backends.items():
                    start = time.monotonic()
                    outputs[backend_name] = function(q, k, v)
                    torch.cuda.synchronize()
                    case["measurements"][backend_name] = {"cold_first_call_wall_s": time.monotonic() - start,
                                                          "blocks_ms": []}
                    for _ in range(5):
                        function(q, k, v)
                torch.cuda.synchronize()
                # Alternate timing order to reduce drift from clock/temperature.
                for block in range(7):
                    for backend_name in (list(backends) if block % 2 == 0 else list(reversed(backends))):
                        function = backends[backend_name]
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize()
                        base = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        start.record()
                        for _ in range(20):
                            function(q, k, v)
                        end.record()
                        end.synchronize()
                        stats = case["measurements"][backend_name]
                        stats["blocks_ms"].append(start.elapsed_time(end) / 20)
                        stats["incremental_peak_allocated_bytes"] = max(stats.get("incremental_peak_allocated_bytes", 0),
                                                                         torch.cuda.max_memory_allocated() - base)
                for stats in case["measurements"].values():
                    stats["median_ms"] = statistics.median(stats["blocks_ms"])
                if include_fa4:
                    a, b = outputs["sdpa"].float(), outputs["fa4"].float()
                    case["accuracy"]["fa4_vs_sdpa_max_abs"] = (a - b).abs().max().item()
                    case["accuracy"]["fa4_vs_sdpa_mean_abs"] = (a - b).abs().mean().item()
                    case["accuracy"]["fa4_vs_sdpa_relative_l2"] = ((a - b).norm() / a.norm()).item()
                    del a, b
                # Full-key FP32 reference on a bounded query/head slice, independent
                # of both fused implementations. No full quadratic score tensor.
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                    reference = torch.nn.functional.scaled_dot_product_attention(
                        q[:, :8, :2].float().transpose(1, 2), k[:, :, :2].float().transpose(1, 2),
                        v[:, :, :2].float().transpose(1, 2)).transpose(1, 2)
                for backend_name, output in outputs.items():
                    sample = output[:, :8, :2].float()
                    case["accuracy"][backend_name + "_vs_fp32_max_abs"] = (sample - reference).abs().max().item()
                    case["accuracy"][backend_name + "_vs_fp32_relative_l2"] = ((sample - reference).norm() / reference.norm()).item()
                if include_fa4:
                    case["kernel_speedup"] = case["measurements"]["sdpa"]["median_ms"] / case["measurements"]["fa4"]["median_ms"]
                report["cases"].append(case)
                print(json.dumps(case), flush=True)
                del q, k, v, outputs, reference, sample
                torch.cuda.empty_cache()
        report["status"] = "completed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
    print(json.dumps(report), flush=True)
    return json.loads(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fa4", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = benchmark(args.fa4)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["status"] == "completed" else 1)
