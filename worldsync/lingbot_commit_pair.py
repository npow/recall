"""Frozen, serial H200 stock/Commit-Forcing research pair; no API calls."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    "reproduction/task-memory-benchmark/GC010-seed42-turn90_wait_return/stock-run.json"
)
CODE = [
    "worldsync/lingbot_commit_pair.py",
    "worldsync/lingbot_commit_forcing.py",
    "worldsync/lingbot_rollout.py",
    "worldsync/lingbot_tuning.py",
    "worldsync/lingbot_native.py",
    "worldsync/lingbot_controls.py",
    "worldsync/lingbot_paths.py",
    "worldsync/lingbot_vae.py",
    "worldsync/matrix_attention.py",
    "worldsync/lingbot_pipeline_reuse.py",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def freeze_plan(root=ROOT):
    old = json.loads((root / REFERENCE).read_text())["run"]
    if old["status"] != "completed" or len(old["chunks"]) != old["max_chunks"]:
        raise ValueError("Incomplete source control")
    task = next(
        v
        for v in json.loads((root / "external/PlayWorld/data/gc/data.json").read_text())
        if v["task_id"] == "GC010"
    )
    image = "external/PlayWorld/data/" + task["image_path"]
    files = CODE + [
        REFERENCE,
        "reproduction/lingbot-weights.json",
        "external/PlayWorld/data/gc/data.json",
        image,
    ]
    plan = {
        "protocol": "lingbot-commit-forcing-gc010-paired-v1",
        "benchmark_eligible": False,
        "arms": ["stock", "commit_forcing"],
        "reference_manifest": REFERENCE,
        "source_revision": old["source_revision"],
        "checkpoint_revision": old["checkpoint_revision"],
        "task_id": "GC010",
        "split": "gc",
        "seed": old["seed"],
        "max_chunks": old["max_chunks"],
        "fixed_actions": old["fixed_actions"],
        "turn_degrees_per_second": old["turn_degrees_per_second"],
        "local_attn_size": 18,
        "sink_size": 6,
        "sampler_timesteps_index": old["sampler_timesteps_index"],
        "sample_shift": old["sample_shift"],
        "text_prompt_mode": "caption",
        "attention_backend": "sdpa",
        "gpu": "H200",
        "torch": "2.6.0+cu124",
        "cuda": "12.4",
        "pipeline_constructor_calls_required": 1,
        "offload_text_encoder": True,
        "cuda_allocator_configuration": "expandable_segments:True",
        "initial_image_sha256": sha(root / image),
        "image_caption_sha256": hashlib.sha256(
            task["image_caption"].encode()
        ).hexdigest(),
        "input_files": {name: sha(root / name) for name in files},
        "old_stock_hardware": old["hardware"],
        "old_stock_pixel_hashes": [v["pixel_sha256"] for v in old["chunks"]],
        "old_reference_comparison": "diagnostic only: old H100/code, incomplete hash provenance",
        "validation_gpu_snapshots": {
            "clean_chunks": [0, 1, 2],
            "all_native_layers": True,
            "description": "Exact current K/V byte comparisons; snapshot memory included in reported peaks",
        },
        "gates": [
            "first three rendered raw hashes exact",
            "first clean cache difference on chunk2",
            "all initial noise blocks and scheduler re-noise draws exact",
            "all native camera audits pass",
            "per-phase key lengths correct",
            "one constructor and complete cleanup between arms",
        ],
    }
    plan["plan_sha256"] = digest(plan)
    return plan


def validate_plan(plan, root=ROOT):
    body = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if digest(body) != plan["plan_sha256"]:
        raise ValueError("Frozen plan content changed")
    if (
        plan["task_id"] != "GC010"
        or plan["seed"] != 42
        or plan["max_chunks"] != 14
        or len(plan["fixed_actions"]) != 14
        or plan["turn_degrees_per_second"] != 45
        or plan["local_attn_size"] != 18
        or plan["sink_size"] != 6
        or plan["source_revision"] != "1895d300d8ac936401689b26389f51cbd36530eb"
        or plan["checkpoint_revision"] != "5c33dd40b213598c418fd25bff30fdbd23fd38a7"
    ):
        raise ValueError("Unexpected frozen native configuration")
    for name, expected in plan["input_files"].items():
        if sha(root / name) != expected:
            raise ValueError("Frozen input changed: " + name)


def verify_pair(summary, runs, plan):
    a, b = runs["stock"], runs["commit_forcing"]
    for run in (a, b):
        if run["status"] != "completed" or len(run["chunks"]) != 14:
            raise ValueError("Incomplete fourteen-chunk arm")
        if len(run["camera_embedding_audit"]) != 14 or not all(
            x["within_tolerance"] for x in run["camera_embedding_audit"]
        ):
            raise ValueError("Native camera audit failed")
    fields = [
        "source_revision",
        "checkpoint_revision",
        "initial_image_sha256",
        "image_caption_sha256",
        "task_id",
        "split",
        "seed",
        "max_chunks",
        "local_attn_size",
        "sink_size",
        "sampler_timesteps_index",
        "sample_shift",
        "fixed_actions",
        "turn_degrees_per_second",
        "hardware",
        "attention_backend",
        "adapter_sha256",
        "controls_sha256",
        "attention_implementation_sha256",
        "cuda_allocator_configuration",
    ]
    matched = {key: a.get(key) == b.get(key) for key in fields}
    required = {
        key: plan[key]
        for key in (
            "source_revision",
            "checkpoint_revision",
            "task_id",
            "split",
            "seed",
            "max_chunks",
            "fixed_actions",
            "turn_degrees_per_second",
            "local_attn_size",
            "sink_size",
            "sampler_timesteps_index",
            "sample_shift",
            "text_prompt_mode",
            "attention_backend",
            "initial_image_sha256",
            "image_caption_sha256",
            "cuda_allocator_configuration",
        )
    }
    required.update(
        protocol_id=plan["protocol"],
        native_fixed_controls=True,
        quantization="none",
        fps=16,
        chunk_size=4,
        max_area=480 * 832,
        generated_frames=221,
        adapter_sha256=plan["input_files"]["worldsync/lingbot_rollout.py"],
        controls_sha256=plan["input_files"]["worldsync/lingbot_controls.py"],
        attention_implementation_sha256=plan["input_files"][
            "worldsync/matrix_attention.py"
        ],
    )
    frozen_matches = {
        arm: all(run.get(key) == value for key, value in required.items())
        and run["hardware"]["torch"] == plan["torch"]
        and run["hardware"]["cuda"] == plan["cuda"]
        and "H200" in run["hardware"]["gpu"]
        and not any(
            run.get(key)
            for key in (
                "pose_retrieval",
                "sink_rephase",
                "kv_retention",
                "decoprune",
                "budget_memory",
                "reanchor",
                "memory_reader",
                "lora",
                "lora_adapter",
            )
        )
        for arm, run in runs.items()
    }
    pixels = [
        x["pixel_sha256"] == y["pixel_sha256"] for x, y in zip(a["chunks"], b["chunks"])
    ]
    rows = {r["arm"]: r for r in summary["arms"]}
    if any(
        len(row["initial_noise"]) != 14 or len(row["renoise"]) != 42
        for row in rows.values()
    ):
        raise ValueError(
            "Expected14 initial noise blocks and42 scheduler noise draws per arm"
        )
    noises = rows["stock"]["initial_noise"] == rows["commit_forcing"]["initial_noise"]
    renoise = rows["stock"]["renoise"] == rows["commit_forcing"]["renoise"]
    changes = summary["clean_cache_comparison"]
    if {(row["chunk"], row["layer"]) for row in changes} != {
        (c, l) for c in range(3) for l in range(40)
    } or len(changes) != 120:
        raise ValueError("Missing exact clean-cache comparisons")
    first = next(
        (
            c
            for c in range(3)
            if any(
                not x["key_equal"] or not x["value_equal"]
                for x in changes
                if x["chunk"] == c
            )
        ),
        None,
    )
    indices = rows["stock"]["cache_indices"] == rows["commit_forcing"]["cache_indices"]
    clean_checks = []
    for arm in plan["arms"]:
        trace = rows[arm]["attention"]["trace"]
        if len(trace) != 14 * 5 * 40:
            raise ValueError("Unexpected native attention/forward count")
        clean_checks.append(
            all(
                x["read_key_tokens"]
                == (
                    min(x["available_key_tokens"], 2 * x["query_tokens"])
                    if arm == "commit_forcing" and x["clean"]
                    else x["available_key_tokens"]
                )
                for x in trace
            )
        )
    proof = {
        "matched_settings": matched,
        "frozen_settings_match": frozen_matches,
        "pixel_hash_equal": pixels,
        "first_three_pixels_exact": all(pixels[:3]),
        "first_different_video_chunk": next(
            (i for i, equal in enumerate(pixels) if not equal), None
        ),
        "initial_noise_exact": noises,
        "renoise_draws_exact": renoise,
        "cache_indices_exact": indices,
        "first_differing_clean_cache_chunk": first,
        "per_phase_read_lengths_correct": all(clean_checks),
        "one_pipeline": summary["constructor_calls"] == 1,
        "old_stock_pixel_hash_equal": [
            x["pixel_sha256"] == y
            for x, y in zip(a["chunks"], plan["old_stock_pixel_hashes"])
        ],
    }
    proof["passed"] = bool(
        all(matched.values())
        and all(frozen_matches.values())
        and all(pixels[:3])
        and noises
        and renoise
        and first == 2
        and indices
        and all(clean_checks)
        and proof["one_pipeline"]
    )
    return proof


def run_pair(plan, output):
    validate_plan(plan)
    import torch

    if (
        str(torch.__version__) != "2.6.0+cu124"
        or torch.version.cuda != "12.4"
        or "H200" not in torch.cuda.get_device_name()
    ):
        raise ValueError("Frozen probe requires H200 Torch2.6/cu124 SDPA")
    import worldsync.lingbot_rollout as rollout
    from worldsync.lingbot_pipeline_reuse import PipelineReuse
    from worldsync.lingbot_commit_forcing import CommitForcing
    import inspect

    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "status": "running",
        "plan_sha256": plan["plan_sha256"],
        "arms": [],
        "clean_cache_comparison": [],
        "validation_snapshot_bytes": 0,
        "benchmark_eligible": False,
    }
    snapshots = {}
    runs = {}
    state = {}
    original_backend = rollout.configure_backend
    original_builder = rollout.build_pipeline

    def save():
        path = output / "summary.tmp"
        path.write_text(json.dumps(summary, indent=2) + "\n")
        path.replace(output / "summary.json")

    def tensor_hash(t):
        return hashlib.sha256(
            t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        ).hexdigest()

    def backend(report, write):
        state.update(report=report, write=write)
        report.update(protocol_id=plan["protocol"], benchmark_eligible=False)
        report["commit_pair"] = {
            "arm": state["row"]["arm"],
            "plan_sha256": plan["plan_sha256"],
            "implementation_sha256": plan["input_files"][
                "worldsync/lingbot_commit_forcing.py"
            ],
            "validation_snapshots_included_in_gpu_peak": True,
        }
        write()
        return original_backend(report, write)

    def build(*args, **kwargs):
        pipeline = original_builder(*args, **kwargs)
        row = state["row"]
        if len(pipeline.model.blocks) != 40:
            raise ValueError("Expected forty native fourteen-billion blocks")
        enabled = row["arm"] == "commit_forcing"
        helper = CommitForcing(pipeline, enabled=enabled, trace_only=not enabled)
        helper.__enter__()
        state["helper"] = helper
        row.update(initial_noise={}, renoise=[], cache_indices=[])
        row["attention"] = helper.metadata()

        def initial(module, positional, keywords):
            chunk = len(state["report"]["chunks"])
            if not helper.clean and str(chunk) not in row["initial_noise"]:
                t = keywords["x"][0]
                row["initial_noise"][str(chunk)] = {
                    "sha256": tensor_hash(t),
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                }

        state["hooks"].append(
            pipeline.model.register_forward_pre_hook(initial, with_kwargs=True)
        )
        for index, block in enumerate(pipeline.model.blocks):
            signature = inspect.signature(block.self_attn.forward)

            def after(
                layer, positional, keywords, result, index=index, signature=signature
            ):
                chunk = len(state["report"]["chunks"])
                if helper.trace[-1]["layer"] != index:
                    raise ValueError("Native self-attention trace order changed")
                helper.trace[-1]["chunk"] = chunk
                if not helper.clean:
                    return
                arguments = signature.bind_partial(*positional, **keywords).arguments
                cache = arguments["kv_cache"]
                tokens = result.shape[1]
                end = int(cache["local_end_index"].item())
                row["cache_indices"].append(
                    {
                        "chunk": chunk,
                        "layer": index,
                        "global_end": int(cache["global_end_index"].item()),
                        "local_end": end,
                        "capacity_tokens": cache["k"].shape[1],
                    }
                )
                if chunk < 3:
                    current = (
                        cache["k"][:, end - tokens : end],
                        cache["v"][:, end - tokens : end],
                    )
                    if not enabled:
                        pair = tuple(t.detach().clone() for t in current)
                        snapshots[chunk, index] = pair
                        summary["validation_snapshot_bytes"] += sum(
                            t.numel() * t.element_size() for t in pair
                        )
                    else:
                        prior = snapshots[chunk, index]
                        summary["clean_cache_comparison"].append(
                            {
                                "chunk": chunk,
                                "layer": index,
                                "key_equal": torch.equal(
                                    current[0].contiguous().view(torch.uint8),
                                    prior[0].view(torch.uint8),
                                ),
                                "value_equal": torch.equal(
                                    current[1].contiguous().view(torch.uint8),
                                    prior[1].view(torch.uint8),
                                ),
                            }
                        )

                if index == 39:
                    save()

            state["hooks"].append(
                block.self_attn.register_forward_hook(after, with_kwargs=True)
            )
        scheduler = pipeline.scheduler
        state["scheduler"] = scheduler
        state["had_add_override"] = "add_noise" in scheduler.__dict__
        state["add_override"] = scheduler.__dict__.get("add_noise")
        original_add = scheduler.add_noise

        def add_noise(original_samples, noise, timesteps, *args, **kwargs):
            row["renoise"].append(
                {
                    "chunk": len(state["report"]["chunks"]),
                    "sha256": tensor_hash(noise),
                    "shape": list(noise.shape),
                    "dtype": str(noise.dtype),
                }
            )
            return original_add(original_samples, noise, timesteps, *args, **kwargs)

        scheduler.add_noise = add_noise
        return pipeline

    rollout.configure_backend = backend
    rollout.build_pipeline = build
    started = time.monotonic()
    try:
        with PipelineReuse(rollout) as worker:
            for arm in plan["arms"]:
                row = {"arm": arm, "status": "running"}
                summary["arms"].append(row)
                state.update(row=row, hooks=[], helper=None, scheduler=None)
                directory = output / arm
                argv = [
                    "--split",
                    "gc",
                    "--task-id",
                    "GC010",
                    "--seed",
                    "42",
                    "--max-chunks",
                    "14",
                    "--native-fixed-controls",
                    "--fixed-actions",
                    *plan["fixed_actions"],
                    "--turn-degrees-per-second",
                    str(plan["turn_degrees_per_second"]),
                    "--local-attn-size",
                    "18",
                    "--sink-size",
                    "6",
                    "--decision-interval",
                    "1",
                    "--offload-text-encoder",
                    "--text-prompt",
                    "caption",
                    "--output",
                    str(directory),
                ]
                try:
                    rollout.main(argv)
                except BaseException:
                    row["status"] = "failed"
                    row["error"] = traceback.format_exc()
                    raise
                finally:
                    for hook in state["hooks"]:
                        hook.remove()
                    if state["helper"] is not None:
                        state["helper"].close()
                        row["attention"] = state["helper"].metadata()
                    if state["scheduler"] is not None:
                        if state["had_add_override"]:
                            state["scheduler"].add_noise = state["add_override"]
                        else:
                            del state["scheduler"].add_noise
                        row["scheduler_binding_restored"] = (
                            "add_noise" in state["scheduler"].__dict__
                        ) == state["had_add_override"] and (
                            not state["had_add_override"]
                            or state["scheduler"].__dict__["add_noise"]
                            is state["add_override"]
                        )
                    row["restored_forward_bindings"] = (
                        worker.pipeline is not None
                        and all(
                            "forward" not in b.self_attn.__dict__
                            for b in worker.pipeline.model.blocks
                        )
                    )
                    summary["constructor_calls"] = worker.constructor_calls
                    save()
                row["cleanup"] = worker.audit()
                if (
                    not row["restored_forward_bindings"]
                    or not row["scheduler_binding_restored"]
                ):
                    raise ValueError("Native forward binding leaked")
                runs[arm] = json.loads((directory / "run.json").read_text())
                row["persistent_worker"] = runs[arm]["persistent_worker"]
                row["status"] = "completed"
                save()
            summary["proof"] = verify_pair(summary, runs, plan)
            if not summary["proof"]["passed"]:
                raise ValueError("Paired proof gate failed; inspect structuredproof")
            summary["status"] = "completed"
    except BaseException:
        summary["status"] = "failed"
        summary["error"] = traceback.format_exc()
        if "row" in state and state["row"].get("status") != "completed":
            state["row"]["status"] = "failed"
            state["row"].setdefault("error", summary["error"])
        raise
    finally:
        snapshots.clear()
        rollout.configure_backend = original_backend
        rollout.build_pipeline = original_builder
        summary["wall_s"] = time.monotonic() - started
        save()
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run_pair(json.loads(a.plan.read_text()), a.output)


if __name__ == "__main__":
    main()
