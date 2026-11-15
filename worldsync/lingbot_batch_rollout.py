"""Run a serial group of native fixed experiments with one pretrained pipeline."""
import argparse
import gc
import json
from pathlib import Path
import re
import sys
import time
import traceback

from worldsync.lingbot_native import SOURCE_REVISION, WEIGHT_REVISION


def group_key(job):
    for token in job["argv"]:
        option = token.split("=", 1)[0]
        if option.startswith("--") and any(full.startswith(option) and full != option
                                           for full in ("--local-attn-size", "--sink-size")):
            raise ValueError("Spell model configuration flags in full for exact batch grouping")
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--local-attn-size", type=int, default=18)
    parser.add_argument("--split", choices=("gc",), required=True)
    parser.add_argument("--sink-size", type=int, default=6)
    parser.add_argument("--native-fixed-controls", action="store_true")
    parser.add_argument("--fixed-actions", nargs="+")
    parser.add_argument("--offload-text-encoder", action="store_true")
    parser.add_argument("--output")
    args, _ = parser.parse_known_args(job["argv"])
    if (not re.fullmatch(r"[A-Za-z0-9_-]+", job["id"]) or args.output is not None
            or not args.native_fixed_controls or not args.fixed_actions or not args.offload_text_encoder):
        raise ValueError("Batch jobs require simple unique ids, native fixed actions, text offload, and no output override")
    if job.get("backend", "sdpa") != "sdpa":
        raise ValueError("Persistent research batches currently validate only the established SDPA runtime")
    return {"source_revision": SOURCE_REVISION, "checkpoint_revision": WEIGHT_REVISION,
            "backend": "sdpa", "torch": "2.6.0+cu124", "dtype": "bfloat16",
            "local_attn_size": args.local_attn_size, "sink_size": args.sink_size,
            "text_encoder_offloaded_after_encoding": True}


def groups(jobs):
    if not jobs:
        raise ValueError("Batch must contain at least one job")
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Job ids must be unique across the entire batch")
    grouped = {}
    for job in jobs:
        key = group_key(job)
        grouped.setdefault(json.dumps(key, sort_keys=True), []).append(job)
    return [{"configuration": json.loads(key), "jobs": rows} for key, rows in grouped.items()]


def verify_run(run, reference=None):
    if (run["status"] != "completed" or len(run["chunks"]) != run["max_chunks"]
            or not run["native_fixed_controls"] or len(run["camera_embedding_audit"]) != run["max_chunks"]
            or not all(row["within_tolerance"] for row in run["camera_embedding_audit"])):
        raise ValueError("Incomplete native rollout or camera audit")
    if reference is None:
        return {"reference_supplied": False}
    fields = ("source_revision", "checkpoint_revision", "task_id", "seed", "split", "fps", "max_chunks",
              "max_area", "chunk_size", "local_attn_size", "sink_size", "fixed_actions", "sampler_timesteps_index",
              "sample_shift", "turn_degrees_per_second", "text_encoder_offloaded_after_encoding", "hardware",
              "initial_image_sha256", "image_caption_sha256")
    if any(run[field] != reference[field] for field in fields):
        raise ValueError("Fresh reference has different generation settings")
    if reference["status"] != "completed" or len(reference["chunks"]) != run["max_chunks"]:
        raise ValueError("Fresh reference is incomplete")
    matches = [a["pixel_sha256"] == b["pixel_sha256"] for a, b in zip(run["chunks"], reference["chunks"])]
    if not all(matches):
        raise ValueError(f"Reused request differs from fresh raw pixels: {matches}")
    return {"reference_supplied": True, "matched_raw_pixel_chunks": len(matches), "all_raw_pixels_identical": True}


def run_batch(plan, output, rollout=None):
    import torch
    from worldsync.lingbot_pipeline_reuse import PipelineReuse
    if rollout is None:
        import worldsync.lingbot_rollout as rollout
    grouped = groups(plan["jobs"])
    if len(grouped) != 1:
        raise ValueError("Each GPU process accepts exactly one initialization group; launcher splits groups")
    if str(torch.__version__) != "2.6.0+cu124":
        raise ValueError("Validated persistent batch runtime requires Torch2.6.0+cu124")
    output.mkdir(parents=True, exist_ok=True)
    summary = {"scope": "serial persistent research batch; no training", "status": "running",
               "configuration": grouped[0]["configuration"], "declared_jobs": len(plan["jobs"]), "jobs": []}
    started = time.monotonic()

    def save():
        temporary = output / "summary.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(output / "summary.json")

    with PipelineReuse(rollout) as worker:
        save()
        for job in plan["jobs"]:
            directory = output / job["id"]
            row = {"id": job["id"], "status": "failed"}
            before = time.monotonic()
            previous_argv = sys.argv
            try:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                sys.argv = ["lingbot_rollout", *job["argv"], "--output", str(directory)]
                rollout.main()
                run = json.loads((directory / "run.json").read_text())
                row["cleanup"] = worker.audit()
                row["parity"] = verify_run(run, job.get("reference"))
                row.update(status="completed", first_chunk_elapsed_s=run["chunks"][0]["elapsed_s"],
                           persistent_worker=run["persistent_worker"], hardware=run["hardware"],
                           generated_chunks=len(run["chunks"]), peak_allocated_bytes=run["peak_allocated_bytes"])
                row["pixel_hashes"] = [chunk["pixel_sha256"] for chunk in run["chunks"]]
            except (Exception, SystemExit):
                row["error"] = traceback.format_exc()
                # A failed request cannot lend potentially partial state to the next.
                worker.discard()
            finally:
                sys.argv = previous_argv
                row["request_wall_s"] = time.monotonic() - before
                summary["jobs"].append(row)
                summary["constructor_calls"] = worker.constructor_calls
                save()
                print(json.dumps({"event": "batch_job_completed", **row}), flush=True)
        summary["worker_wall_s"] = time.monotonic() - started
        summary["status"] = "completed" if all(row["status"] == "completed" for row in summary["jobs"]) else "failed"
        save()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_batch(json.loads(args.jobs.read_text()), args.output)
    raise SystemExit(0 if summary["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
