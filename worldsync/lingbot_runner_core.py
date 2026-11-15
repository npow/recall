"""Job execution for the persistent Modal runner (and the matching one-shot reference).

A job is a plain JSON dict. Either give `argv` (the exact CLI arguments of
`worldsync.lingbot_rollout`, minus `--output`) or the friendly fields below.

    id                 label used in the result directory name
    path | actions     named fixed path from worldsync.lingbot_paths.PATHS, or an explicit action list
    task_id, split     default GC007, gc
    seed               default 42
    cache, sink        rolling window and sink in latent frames (default 18, 6)
    turn_rate          degrees per second, default 60
    env                extra environment variables, e.g. WORLDSYNC_TURN_DEG
    offload_text_encoder, allocator_expandable   lossless H100 memory savings
    extra              extra rollout CLI arguments, e.g. ["--budget-memory", "--budget-retrieved", "10"]
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def job_to_argv(job, output):
    if job.get("argv"):
        argv = list(job["argv"])
        argv += ["--output", str(output)]
        return argv
    from worldsync.lingbot_paths import PATHS
    actions = list(job["actions"]) if job.get("actions") else list(PATHS[job["path"]])
    argv = ["--split", job.get("split", "gc"), "--task-id", job.get("task_id", "GC007"),
            "--seed", str(job.get("seed", 42)), "--max-chunks", str(len(actions)),
            "--native-fixed-controls", "--fixed-actions", *actions, "--output", str(output)]
    if job.get("turn_rate", 60) != 60:
        argv += ["--turn-degrees-per-second", str(job["turn_rate"])]
    cache, sink = job.get("cache", 18), job.get("sink", 6)
    if (cache, sink) != (18, 6):
        argv += ["--controlled-ablation", "--local-attn-size", str(cache), "--sink-size", str(sink)]
    if job.get("offload_text_encoder"):
        argv.append("--offload-text-encoder")
    argv += list(job.get("extra", []))
    return argv


def job_identifier(job):
    return f"{job.get('id') or 'job'}-{time.time_ns()}"


class RunnerCore:
    """Runs jobs inside one warm process. Construct once per container."""

    def __init__(self, work_root="/work", results_dir="/results", commit=None, defaults=None, before_job=None):
        from worldsync.lingbot_runtime import WarmRuntime
        self.runtime = WarmRuntime()
        self.work_root = Path(work_root)
        self.results_dir = Path(results_dir)
        self.commit = commit
        self.defaults = dict(defaults or {})
        self.before_job = before_job
        self.fingerprint = None

    def run(self, job, persist=True):
        from worldsync import lingbot_rollout
        job = {**self.defaults, **job}
        if self.before_job:
            self.before_job(job)
        identifier = job_identifier(job)
        output = self.work_root / "outputs" / identifier
        argv = job_to_argv(job, output)
        saved_env = {key: os.environ.get(key) for key in job.get("env", {})}
        os.environ.update({key: str(value) for key, value in job.get("env", {}).items()})
        started = time.monotonic()
        error = None
        report = None
        try:
            report = lingbot_rollout.main(argv, runtime=self.runtime)
        except SystemExit as exit_request:  # argparse error or parser.error()
            error = f"rollout rejected the job (exit {exit_request.code}); argv={argv}"
        except BaseException as failure:
            error = f"{type(failure).__name__}: {failure}\n{traceback.format_exc()[-2000:]}"
        finally:
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            self.runtime.between_jobs()
        elapsed = time.monotonic() - started
        fingerprint = self.runtime.fingerprint()
        weights_unchanged = None
        if fingerprint is not None:
            if self.fingerprint is None:
                self.fingerprint = fingerprint
            weights_unchanged = fingerprint == self.fingerprint
        run_json = output / "run.json"
        if run_json.exists():
            report = json.loads(run_json.read_text())
        if persist and output.exists() and self.results_dir:
            shutil.copytree(output, self.results_dir / identifier)
            if self.commit:
                self.commit()
        return {"id": job.get("id"), "result_directory": identifier, "argv": argv, "error": error,
                "status": (report or {}).get("status", "failed" if error else "unknown"),
                "chunk_pixel_sha256": [chunk["pixel_sha256"] for chunk in (report or {}).get("chunks", [])],
                "job_seconds": elapsed, "weights_unchanged": weights_unchanged,
                "warm_runtime": (report or {}).get("warm_runtime"),
                "peak_allocated_bytes": (report or {}).get("peak_allocated_bytes"),
                "run": report}


def run_oneshot(job, work_root="/work", results_dir="/results", commit=None):
    """Reference path: a fresh `python -m worldsync.lingbot_rollout` process, so the model reloads."""
    identifier = job_identifier(job)
    output = Path(work_root) / "outputs" / identifier
    argv = job_to_argv(job, output)
    env = dict(os.environ)
    env.update({key: str(value) for key, value in job.get("env", {}).items()})
    if job.get("allocator_expandable"):
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    started = time.monotonic()
    outcome = subprocess.run([sys.executable, "-u", "-m", "worldsync.lingbot_rollout", *argv], cwd=work_root, env=env)
    elapsed = time.monotonic() - started
    report = None
    if (output / "run.json").exists():
        report = json.loads((output / "run.json").read_text())
    if output.exists() and results_dir:
        shutil.copytree(output, Path(results_dir) / identifier)
        if commit:
            commit()
    error = None if outcome.returncode == 0 else f"one-shot exited {outcome.returncode}"
    return {"id": job.get("id"), "result_directory": identifier, "argv": argv, "error": error,
            "status": (report or {}).get("status", "failed" if error else "unknown"),
            "chunk_pixel_sha256": [chunk["pixel_sha256"] for chunk in (report or {}).get("chunks", [])],
            "job_seconds": elapsed, "peak_allocated_bytes": (report or {}).get("peak_allocated_bytes"), "run": report}
