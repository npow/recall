"""Complete matched native rollouts for SDPA26, SDPA29 and opt-in FA4."""
import json
from pathlib import Path
import statistics
import sys

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if modal.is_local():
    from scripts.modal_lingbot_attention_benchmark import fa4_runtime_image
    from scripts.modal_lingbot_benchmark import weights, results
    image = (fa4_runtime_image
             .run_commands("/opt/fa4/bin/python -m pip install torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128",
                           "/opt/fa4/bin/python -m pip install transformers==4.51.3 tokenizers==0.21.4 scipy==1.15.3 diffusers==0.37.0 huggingface_hub==0.36.2 hf_xet accelerate safetensors Pillow einops easydict ftfy sentencepiece tqdm imageio imageio-ffmpeg opencv-python-headless==4.11.0.86")
             .add_local_dir(ROOT / "worldsync", "/work/worldsync", ignore=["__pycache__"])
             .add_local_file(ROOT / "reproduction/lingbot-weights.json", "/work/reproduction/lingbot-weights.json")
             .add_local_file(ROOT / "external/PlayWorld/data/gc/data.json", "/work/external/PlayWorld/data/gc/data.json")
             .add_local_dir(ROOT / "external/PlayWorld/data/gc/images", "/work/external/PlayWorld/data/gc/images"))
else:
    image = None
    weights = modal.Volume.from_name("worldsync-lingbot-weights")
    results = modal.Volume.from_name("worldsync-lingbot-results")

app = modal.App("worldsync-lingbot-runtime-benchmark")


@app.function(image=image, gpu="H100", cpu=8, memory=131072, timeout=1800, max_containers=3,
              volumes={"/cache": weights, "/results": results})
def generate(mode: str):
    import os
    import shutil
    import subprocess
    import time

    if mode not in {"sdpa26", "sdpa29", "fa4"}:
        raise ValueError("Unknown runtime mode")
    weights.reload()
    target = Path("/work/external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast")
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        target.symlink_to("/cache/14b-causal-fast")
    identifier = f"runtime-{mode}-GC007-wait10-seed42-{time.time_ns()}"
    output = Path("/work/outputs") / identifier
    executable = sys.executable if mode == "sdpa26" else "/opt/fa4/bin/python"
    command = [executable, "-u", "-m", "worldsync.lingbot_rollout", "--split", "gc",
               "--task-id", "GC007", "--seed", "42", "--max-chunks", "10",
               "--native-fixed-controls", "--fixed-actions", *(["WAIT"] * 10),
               "--offload-text-encoder", "--output", str(output)]
    environment = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                       WORLDSYNC_LINGBOT_ATTENTION="fa4" if mode == "fa4" else "sdpa")
    try:
        outcome = subprocess.run(command, cwd="/work", env=environment)
    finally:
        if output.exists():
            shutil.copytree(output, Path("/results") / identifier)
            results.commit()
    if outcome.returncode:
        return {"mode": mode, "status": "failed", "result_directory": identifier,
                "run": json.loads((output / "run.json").read_text()) if (output / "run.json").exists() else None}
    return {"mode": mode, "status": "completed", "result_directory": identifier,
            "run": json.loads((output / "run.json").read_text())}


@app.local_entrypoint()
def main():
    output = ROOT / "reproduction/lingbot-runtime-benchmark"
    output.mkdir(exist_ok=True)
    pending = [mode for mode in ("sdpa26", "sdpa29", "fa4")
               if not (output / (mode + ".json")).exists()]
    failures = []
    for index, result in enumerate(generate.map(pending, return_exceptions=True)):
        if isinstance(result, BaseException):
            failures.append({"mode": pending[index], "error": str(result)})
            continue
        (output / (result["mode"] + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    if failures:
        (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise RuntimeError("Runtime generation failed; all returned artifacts were retained")
    reports = {mode: json.loads((output / (mode + ".json")).read_text()) for mode in ("sdpa26", "sdpa29", "fa4")}
    matched_fields = ("source_revision", "checkpoint_revision", "task_id", "split", "seed", "fps",
                      "max_chunks", "max_area", "chunk_size", "local_attn_size", "sink_size",
                      "sampler_timesteps_index", "sample_shift", "fixed_actions", "controls_sha256",
                      "adapter_sha256", "text_encoder_offloaded_after_encoding", "cuda_allocator_configuration",
                      "initial_image_sha256", "image_caption_sha256", "turn_degrees_per_second")
    rows = {}
    reference = reports["sdpa29"]["run"]
    for mode, result in reports.items():
        run = result["run"]
        if result["status"] != "completed" or run["status"] != "completed" or len(run["chunks"]) != 10:
            raise ValueError(f"Incomplete runtime case {mode}; artifacts retained")
        if any(run[field] != reference[field] for field in matched_fields):
            raise ValueError(f"Unmatched runtime case {mode}")
        if (run["hardware"]["gpu"] != reference["hardware"]["gpu"] or len(run["camera_embedding_audit"]) != 10
                or not all(row["within_tolerance"] for row in run["camera_embedding_audit"])):
            raise ValueError(f"Hardware or native camera audit differs for {mode}")
        intervals = [run["chunks"][i]["elapsed_s"] - run["chunks"][i - 1]["elapsed_s"] for i in range(6, 10)]
        rows[mode] = {"result_directory": result["result_directory"], "hardware": run["hardware"],
                      "attention_backend": run["attention_backend"], "warmed_chunk_intervals_s": intervals,
                      "warmed_mean_s": statistics.mean(intervals), "warmed_median_s": statistics.median(intervals),
                      "first_chunk_elapsed_s": run["chunks"][0]["elapsed_s"],
                      "peak_allocated_bytes": run["peak_allocated_bytes"],
                      "pixel_hashes_identical_to_sdpa29": [a["pixel_sha256"] == b["pixel_sha256"]
                                                            for a, b in zip(run["chunks"], reference["chunks"])]}
    if reports["fa4"]["run"]["hardware"] != reference["hardware"]:
        raise ValueError("Primary FA4/SDPA comparison must use identical Torch/CUDA hardware")
    summary = {"scope": "ten WAIT chunks, native controls; speed diagnostic, not task quality",
               "status": "completed", "matched_settings": {field: reference[field] for field in matched_fields},
               "results": rows, "fa4_speedup_vs_same_stack_sdpa": rows["sdpa29"]["warmed_mean_s"] / rows["fa4"]["warmed_mean_s"],
               "fa4_speedup_vs_stock_stack": rows["sdpa26"]["warmed_mean_s"] / rows["fa4"]["warmed_mean_s"],
               "caveat": "One container per mode; warmed intervals include decoding and transport. Different kernels can change BF16 rounding and subsequent generation. No scored-quality equivalence claim."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
