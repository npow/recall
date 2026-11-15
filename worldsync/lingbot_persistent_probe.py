"""Experimental factory reuse across complete rollouts; no production worker API."""
import argparse
import copy
import gc
import json
from pathlib import Path
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import worldsync.lingbot_rollout as rollout

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    original_backend = rollout.configure_backend
    pool = {}
    summary = {"scope": "persistent pipeline throughput/state-isolation diagnostic; stock Torch2.6 SDPA",
               "status": "running", "episodes": [], "constructor_calls": 0,
               "sequence": ["cold-memory42", "reuse-plain43", "reuse-memory42", "reuse-plain42"],
               "training": False, "text_cache_policy": "clear on reuse; recompute native prompt context and retain original encoder placement"}
    started = time.monotonic()

    def save():
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def versions(pipeline):
        return tuple((name, int(parameter._version)) for name, parameter in pipeline.model.named_parameters())

    def hook_count(pipeline):
        return sum(len(module._forward_pre_hooks) + len(module._forward_hooks)
                   for module in pipeline.model.modules())

    def configured(report, persist):
        upstream = original_backend(report, persist)
        if "constructor" not in pool:
            pool["constructor"] = upstream.WanI2VCausal

        def construct(*positional, **kwargs):
            signature = (tuple(repr(value) for value in positional),
                         tuple(sorted((key, repr(value)) for key, value in kwargs.items())))
            before = time.monotonic()
            reused = "pipeline" in pool
            if reused:
                if signature != pool["signature"]:
                    raise ValueError("Probe only reuses identical pipeline initialization settings")
                pipeline = pool["pipeline"]
                if hook_count(pipeline) != pool["initial_hooks"]:
                    raise ValueError("Hooks leaked from a previous request")
                if versions(pipeline) != pool["parameter_versions"]:
                    raise ValueError("DiT parameters changed across requests")
                # Preserve fresh-request prompt encoding and GPU residency. This
                # probe optimizes checkpoint loading, not prompt-cache behavior.
                pipeline.clear_text_cache()
                pipeline.vae.model.clear_cache()
                report["checkpoint_loading"] = copy.deepcopy(pool["loading_info"])
            else:
                pipeline = pool["constructor"](*positional, **kwargs)
                pool.update(pipeline=pipeline, signature=signature,
                            loading_info=copy.deepcopy(report["checkpoint_loading"]),
                            parameter_versions=versions(pipeline), initial_hooks=hook_count(pipeline))
                summary["constructor_calls"] += 1
            report["persistent_worker"] = {"pipeline_reused": reused,
                "constructor_or_reuse_wall_s": time.monotonic() - before,
                "text_cache_entries_before_generation": len(pipeline._t5_cache),
                "encoder_device_before_generation": str(next(pipeline.text_encoder.model.parameters()).device)}
            persist()
            return pipeline

        upstream.WanI2VCausal = construct
        return upstream

    rollout.configure_backend = configured
    save()
    try:
        for name, seed, memory in (("cold-memory42", 42, True), ("reuse-plain43", 43, False),
                                   ("reuse-memory42", 42, True), ("reuse-plain42", 42, False)):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            directory = output / name
            argv = ["lingbot_rollout", "--split", "gc", "--task-id", "GC007", "--seed", str(seed),
                    "--max-chunks", "10", "--native-fixed-controls", "--fixed-actions", *(["WAIT"] * 10),
                    "--offload-text-encoder", "--output", str(directory)]
            if memory:
                argv.append("--pose-retrieval")
            previous_argv = sys.argv
            episode_start = time.monotonic()
            try:
                sys.argv = argv
                rollout.main()
            finally:
                sys.argv = previous_argv
            run = json.loads((directory / "run.json").read_text())
            pipeline = pool["pipeline"]
            cleanup = {"remaining_model_hooks": hook_count(pipeline),
                       "dit_parameter_versions_unchanged": versions(pipeline) == pool["parameter_versions"],
                       "encoder_device_after_request": str(next(pipeline.text_encoder.model.parameters()).device),
                       "text_cache_entries_after_request": len(pipeline._t5_cache)}
            row = {"name": name, "seed": seed, "memory": memory, "status": run["status"],
                   "total_request_wall_s": time.monotonic() - episode_start,
                   "first_chunk_elapsed_s": run["chunks"][0]["elapsed_s"],
                   "persistent_worker": run["persistent_worker"], "cleanup": cleanup,
                   "pixel_hashes": [chunk["pixel_sha256"] for chunk in run["chunks"]]}
            summary["episodes"].append(row)
            save()
            if (run["status"] != "completed" or len(run["chunks"]) != 10
                    or len(run["camera_embedding_audit"]) != 10
                    or not all(item["within_tolerance"] for item in run["camera_embedding_audit"])
                    or cleanup["remaining_model_hooks"] != pool["initial_hooks"]
                    or not cleanup["dit_parameter_versions_unchanged"]
                    or cleanup["encoder_device_after_request"] != "cpu"):
                raise ValueError("Persistent request failed completion or state-cleanup audit")
        episodes = {row["name"]: row for row in summary["episodes"]}
        fresh = json.loads(Path("/work/reproduction/fresh-sdpa26.json").read_text())["run"]
        summary["memory42_repeat_pixel_identical"] = (episodes["cold-memory42"]["pixel_hashes"] == episodes["reuse-memory42"]["pixel_hashes"])
        summary["plain42_matches_fresh_external_control"] = (episodes["reuse-plain42"]["pixel_hashes"] == [chunk["pixel_sha256"] for chunk in fresh["chunks"]])
        summary["different_seed_changes_output"] = episodes["reuse-plain43"]["pixel_hashes"] != episodes["reuse-plain42"]["pixel_hashes"]
        summary["fresh_plain42_control_wall_s"] = fresh["elapsed_s"]
        summary["fresh_plain42_control_hardware"] = fresh["hardware"]
        if (summary["constructor_calls"] != 1 or not summary["memory42_repeat_pixel_identical"]
                or not summary["plain42_matches_fresh_external_control"] or not summary["different_seed_changes_output"]):
            raise ValueError("Persistent reuse did not preserve independent-request generation")
        summary["status"] = "completed"
    except BaseException:
        summary["status"] = "failed"
        summary["error"] = traceback.format_exc()
        raise
    finally:
        summary["worker_wall_s"] = time.monotonic() - started
        save()
        rollout.configure_backend = original_backend
        if "constructor" in pool:
            import wan.image2video as upstream
            upstream.WanI2VCausal = pool["constructor"]


if __name__ == "__main__":
    main()
