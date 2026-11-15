"""Serial isolated NanoKV episodes using the unchanged native rollout.

Fixed jobs use the existing controlled-ablation gate. Adaptive jobs retain the
existing hillclimb gate and controller; this module does not unlock either.
"""

import argparse
import hashlib
import json
import math
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_RUNNING = False
_FIELDS = {
    "id",
    "split",
    "task_id",
    "seed",
    "max_chunks",
    "capacity_frames",
    "fixed_actions",
    "text_prompt",
    "turn_degrees_per_second",
    "stream_player_actions",
}
_ACTIONS = {"W", "A", "S", "D", "LEFT", "RIGHT", "UP", "DOWN", "WAIT"}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_spec(spec):
    """Reject ambiguous or silently ignored controls before model/API work."""
    if not isinstance(spec, dict) or set(spec) - _FIELDS:
        raise ValueError("Unknown NanoKV episode fields")
    config = {
        "seed": 42,
        "fixed_actions": None,
        "text_prompt": "caption",
        "turn_degrees_per_second": 60,
        "stream_player_actions": False,
        **spec,
    }
    if "id" in config and (not isinstance(config["id"], str) or not config["id"]):
        raise ValueError("Episode IDs must be nonempty strings")
    if config.get("split") not in {"gc", "if", "insight", "outsight"}:
        raise ValueError("Invalid dataset split")
    if not isinstance(config.get("task_id"), str) or not config["task_id"].strip():
        raise ValueError("A nonempty task_id is required")
    for name in ("seed", "max_chunks", "capacity_frames"):
        if type(config.get(name)) is not int:
            raise ValueError(f"{name} must be an integer")
    maximum = 60 if config["split"] == "insight" else 40
    if not 1 <= config["max_chunks"] <= maximum:
        raise ValueError("Chunk budget exceeds native split limits")
    capacity = config["capacity_frames"]
    if capacity < 10 or (capacity - 6) % 4:
        raise ValueError(
            "Capacity must preserve six sinks and four-frame rolling blocks"
        )
    if config["text_prompt"] not in {"caption", "prompt", "both"}:
        raise ValueError("Invalid text_prompt")
    rate = config["turn_degrees_per_second"]
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 < rate <= 90:
        raise ValueError("Turn rate must be finite in (0,90]")
    if type(config["stream_player_actions"]) is not bool:
        raise ValueError("stream_player_actions must be Boolean")
    actions = config["fixed_actions"]
    if actions is not None:
        if (
            not isinstance(actions, list)
            or len(actions) != config["max_chunks"]
            or any(not isinstance(a, str) or a not in _ACTIONS for a in actions)
        ):
            raise ValueError("Fixed actions must cover every native chunk")
        if config["stream_player_actions"]:
            raise ValueError("Streaming is only applicable to player-driven jobs")
        if config["split"] == "insight" and any(a != "WAIT" for a in actions):
            raise ValueError("Insight is stationary; non-WAIT actions would be ignored")
        config["fixed_actions"] = list(actions)
    return config


def episode_argv(config, output):
    argv = [
        "--split",
        config["split"],
        "--task-id",
        config["task_id"],
        "--seed",
        str(config["seed"]),
        "--max-chunks",
        str(config["max_chunks"]),
        "--local-attn-size",
        str(config["capacity_frames"]),
        "--sink-size",
        "6",
        "--offload-text-encoder",
        "--text-prompt",
        config["text_prompt"],
        "--turn-degrees-per-second",
        str(config["turn_degrees_per_second"]),
        "--output",
        str(output),
    ]
    if config["fixed_actions"] is not None:
        argv += [
            "--controlled-ablation",
            "--native-fixed-controls",
            "--fixed-actions",
            *config["fixed_actions"],
        ]
    elif config["stream_player_actions"]:
        argv += ["--stream-player-actions"]
    return argv


def _save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_episode(
    spec: dict, output: Path, *, codec_path: Path, source_root: Path
) -> dict:
    """Run one request; safe inside PipelineReuse for serial equal-capacity jobs.

    Returns output/summary.json's content. Original videos/run.json live in
    output/episode. Failed requests retain both manifests and raise to callers.
    """
    global _RUNNING
    if _RUNNING:
        raise RuntimeError("NanoKV episode wrappers are serial and cannot overlap")
    config = normalize_spec(spec)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    from worldsync import lingbot_rollout as rollout
    from worldsync.lingbot_nanokv import CODEC_SHA256, NativeNanoKV
    from worldsync.lingbot_pipeline_reuse import hook_count, parameter_versions

    original_build = rollout.build_pipeline
    state = {}
    summary = {
        "status": "running",
        "scope": "Isolated NanoKV research episode; not stock or published-score reproduction",
        "benchmark_eligible": False,
        "config": config,
        "codec_sha256": CODEC_SHA256,
        "wrapper_sha256": _sha(Path(__file__)),
        "run_path": "episode/run.json",
        "diagnostic_mode": "native-fixed"
        if config["fixed_actions"] is not None
        else "adaptive-or-stationary",
        "adaptive_uses_existing_hillclimb_gate": config["fixed_actions"] is None,
        "argv": episode_argv(config, output / "episode"),
    }
    started = time.monotonic()

    def build(*args, **kwargs):
        if state:
            raise RuntimeError("One episode may attach only one pipeline")
        pipeline = original_build(*args, **kwargs)
        state["pipeline"] = pipeline
        state["versions"] = parameter_versions(pipeline)
        state["hooks"] = hook_count(pipeline)
        adapter = NativeNanoKV(
            pipeline,
            source_root=source_root,
            codec_path=codec_path,
            capacity_frames=config["capacity_frames"],
            identity=False,
        )
        state["adapter"] = adapter
        adapter.__enter__()
        return pipeline

    _RUNNING = True
    failure = None
    try:
        _save(output / "summary.json", summary)
        rollout.build_pipeline = build
        report = rollout.main(summary["argv"])
        if not isinstance(report, dict) or report.get("status") != "completed":
            raise ValueError("Native rollout did not complete")
        if "pipeline" not in state:
            raise ValueError("Native rollout did not build/attach the NanoKV pipeline")
        chunks = report.get("chunks", [])
        if not chunks or len(chunks) > config["max_chunks"]:
            raise ValueError("Native rollout has an invalid generated chunk count")
        if config["fixed_actions"] is not None:
            audits = report.get("camera_embedding_audit", [])
            if (
                len(chunks) != config["max_chunks"]
                or len(audits) != len(chunks)
                or not all(row.get("within_tolerance") is True for row in audits)
            ):
                raise ValueError(
                    "Fixed NanoKV episode requires complete native camera audits"
                )
            summary["all_native_camera_audits_passed"] = True
        summary["completed_chunks"] = len(chunks)
        summary["raw_pixel_hashes"] = [chunk["pixel_sha256"] for chunk in chunks]
        summary["status"] = "completed"
    except BaseException as error:
        failure = error
        summary["status"] = "failed"
        summary["error"] = traceback.format_exc()
        raise
    finally:
        rollout.build_pipeline = original_build
        cleanup_errors = []
        adapter = state.get("adapter")
        try:
            if adapter is not None:
                adapter.close()
                summary["adapter"] = adapter.metadata()
            pipeline = state.get("pipeline")
            if pipeline is not None:
                summary["base_parameter_versions_unchanged"] = (
                    parameter_versions(pipeline) == state["versions"]
                )
                summary["model_hooks_restored"] = hook_count(pipeline) == state["hooks"]
                if (
                    not summary["base_parameter_versions_unchanged"]
                    or not summary["model_hooks_restored"]
                ):
                    raise ValueError(
                        "Pipeline state changed across isolated NanoKV request"
                    )
            if adapter is not None and not summary["adapter"].get("closed"):
                raise ValueError("NanoKV adapter did not close")
        except BaseException:  # noqa: BLE001 -- retain cleanup evidence without masking the original failure
            cleanup_errors.append(traceback.format_exc())
        if cleanup_errors:
            summary["status"] = "failed"
            summary["cleanup_errors"] = cleanup_errors
        summary["wall_s"] = time.monotonic() - started
        try:
            run_path = output / "episode/run.json"
            if run_path.exists():
                report = json.loads(run_path.read_text())
                if summary["status"] != "completed":
                    report["native_rollout_status"] = report.get("status")
                    report["status"] = "failed"
                report["model"] = (
                    "LingBot-World2 14B causal-fast + isolated NanoKV compression"
                )
                report["stock_knobs"] = False
                report["published_score_reproduced"] = False
                report["nanokv_episode"] = {
                    key: summary[key]
                    for key in (
                        "status",
                        "config",
                        "codec_sha256",
                        "wrapper_sha256",
                        "diagnostic_mode",
                        "benchmark_eligible",
                    )
                }
                report["nanokv_episode"]["adapter"] = summary.get("adapter")
                _save(run_path, report)
            _save(output / "summary.json", summary)
        finally:
            _RUNNING = False
        if cleanup_errors and failure is None:
            raise RuntimeError("NanoKV cleanup/state audit failed; see summary.json")
    return summary


def validate_plan(plan, root=ROOT):
    """Check the complete frozen plan and every repository-relative input."""
    if not isinstance(plan, dict):
        raise TypeError("Frozen plan must be a dictionary")
    root = Path(root).resolve()
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != plan.get("plan_sha256"):
        raise ValueError("NanoKV episode plan checksum mismatch")
    inputs = plan.get("input_files")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("Frozen plans require input hashes")
    for name, expected in inputs.items():
        resolved = (root / name).resolve()
        if Path(name).is_absolute() or not resolved.is_relative_to(root):
            raise ValueError("Frozen input paths must remain in this repository")
        if _sha(resolved) != expected:
            raise ValueError("Frozen input changed: " + name)
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Frozen plans require jobs")
    if any(not isinstance(job, dict) for job in jobs):
        raise ValueError("Frozen jobs must be dictionaries")
    ids = [job.get("id") for job in jobs]
    if any(not isinstance(item, str) or not item for item in ids) or len(
        set(ids)
    ) != len(ids):
        raise ValueError("Frozen job IDs must be unique nonempty strings")
    for job in jobs:
        normalize_spec(job)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = validate_plan(json.loads(args.plan.read_text()))
    selected = [job for job in plan["jobs"] if job["id"] == args.job_id]
    if len(selected) != 1:
        raise ValueError("Job ID not present in frozen plan")
    spec = selected[0]
    from worldsync.lingbot_nanokv import CODEC_SHA256

    if (
        plan.get("codec_sha256", CODEC_SHA256) != CODEC_SHA256
        or _sha(args.codec) != CODEC_SHA256
    ):
        raise ValueError("Frozen NanoKV codec changed")
    return run_episode(
        spec, args.output, codec_path=args.codec, source_root=ROOT / "external/NanoKV"
    )


if __name__ == "__main__":
    main()
