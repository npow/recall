"""Warm adaptive-episode worker for a plain GPU machine (Lambda Cloud). Run ON the instance, one process per GPU:

    cd ~/work && CUDA_VISIBLE_DEVICES=0 .venv-lambda/bin/python scripts/lambda_worker.py \
        --jobs jobs.jsonl --shard 0 --shards 1 [--canary]

Same engine as the Modal adaptive runner: one warm RunnerCore (model loaded once), the same
queue logic (worldsync.lingbot_episode_queue.drain_loop: resume, transient retries, progress
records) and the same result layout under ~/work/results/adaptive/<label>/... . Jobs are the
dicts built by lingbot_episode_queue.episode_job (raw rollout argv), one JSON per line. A worker
takes the lines whose index % shards == shard, so several machines share one job list without
coordination. ANTHROPIC_API_KEY must be in the environment (it is passed over SSH, never stored).
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # as the Modal H100 class
os.environ.setdefault("WORLDSYNC_LINGBOT_ATTENTION", "sdpa")

WARMUP_JOB = {"id": "warmup", "actions": ["WAIT"], "task_id": "GC007", "offload_text_encoder": True}
CANARY_JOB = {"id": "canary", "path": "turn_return", "task_id": "GC007", "seed": 42, "offload_text_encoder": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--canary", action="store_true", help="run the fixed 8-chunk stock canary before and after")
    parser.add_argument("--canary-only", action="store_true")
    parser.add_argument("--plain", metavar="LABEL", help="dev mode: each job is any RunnerCore job ({id, argv} or "
                        "friendly fields); results go to results/dev/LABEL/<id>/; finished ids are skipped")
    parser.add_argument("--plain-retries", type=int, default=0, help="extra attempts per job in --plain mode")
    args = parser.parse_args()
    from worldsync import lingbot_episode_queue as q
    from worldsync.lingbot_runner_core import RunnerCore
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    core = RunnerCore(work_root=str(ROOT), results_dir=str(results))
    warm = core.run(WARMUP_JOB, persist=False)
    print(json.dumps({"event": "warmup", "status": warm["status"], "seconds": warm["job_seconds"],
                      "error": (warm.get("error") or "")[:300]}), flush=True)

    def canary(tag):
        outcome = core.run({**CANARY_JOB, "result_name": f"canary/{tag}-{time.time_ns()}"})
        print(json.dumps({"event": "canary", "tag": tag, "status": outcome["status"], "seconds": outcome["job_seconds"],
                          "chunk_pixel_sha256": outcome["chunk_pixel_sha256"],
                          "peak_allocated_bytes": outcome["peak_allocated_bytes"],
                          "error": (outcome.get("error") or "")[:300]}), flush=True)
        return outcome["chunk_pixel_sha256"]

    hashes = []
    if args.canary or args.canary_only:
        hashes.append(canary("before"))
    if args.canary_only:
        return
    jobs = [json.loads(line) for line in args.jobs.read_text().splitlines() if line.strip()]
    mine = iter([job for index, job in enumerate(jobs) if index % args.shards == args.shard])
    if args.plain:
        return plain(core, list(mine), args.plain, results, retries=args.plain_retries)

    def list_directory(path):
        try:
            return sorted(os.listdir(results / path))
        except OSError:
            return []

    def read_json(path):
        return json.loads((results / path).read_text())

    def run_job(job, attempt):
        name = q.attempt_name(job, attempt)
        stale = results / name
        if stale.exists():  # a partial directory from an interrupted attempt would break copytree
            shutil.rmtree(stale)
        outcome = core.run({**job, "result_name": name})
        shutil.rmtree(ROOT / "outputs" / name, ignore_errors=True)
        run = outcome.pop("run", None) or {}
        outcome.update(generated_seconds=run.get("generated_seconds"), termination=run.get("termination"),
                       chunks=len(run.get("chunks", [])))
        print(json.dumps({"event": "attempt", "id": job["id"], "attempt": attempt, "status": outcome["status"],
                          "seconds": round(outcome["job_seconds"], 1), "chunks": outcome["chunks"],
                          "error": (outcome.get("error") or "")[:300]}), flush=True)
        return outcome

    def record(kind, job, payload):
        directory = results / "adaptive" / job["label"] / kind
        directory.mkdir(parents=True, exist_ok=True)
        payload = {**payload, "arm": job["arm"], "split": job["split"], "task_id": job["task_id"], "seed": job["seed"],
                   "base": job["base"], "host": os.uname().nodename, "shard": args.shard}
        (directory / f"{job['id']}.json").write_text(json.dumps(payload, indent=1) + "\n")

    summary = q.drain_loop(lambda: next(mine, None), run_job, lambda job: q.episode_done(job, list_directory, read_json),
                           record, max_attempts=args.max_attempts, max_episodes=args.max_episodes)
    if args.canary:
        hashes.append(canary("after"))
        summary["canary_identical_after_episodes"] = hashes[0] == hashes[1] and bool(hashes[0])
    summary["warm_jobs_run"] = core.runtime.jobs_run
    print(json.dumps({"event": "summary", **summary}), flush=True)


def plain(core, jobs, label, results, retries=0):
    """Jobs from any branch: one result directory per job id, resumable, with kept failed attempts.

    results/dev/<label>/<id>/                    the completed attempt (rollout.mp4, run.json, ...)
    results/dev/<label>-attempts/<id>__a<k>/      every failed attempt, kept
    results/dev/<label>-failed/<id>.json          written when a job has used all its attempts
    A job gets 1 + retries attempts in total, counted on disk, so a restarted worker never retries forever.
    A fatal CUDA error ends the process (exit 3); the tmux wrapper restarts it with a fresh model.
    """
    from worldsync import lingbot_episode_queue as q
    attempts_root = results / "dev" / f"{label}-attempts"
    failed_root = results / "dev" / f"{label}-failed"
    done = failed = 0
    for job in jobs:
        name = f"dev/{label}/{job['id']}"
        target = results / name
        try:
            if json.loads((target / "run.json").read_text()).get("status") == "completed":
                print(json.dumps({"event": "skip", "id": job["id"]}), flush=True)
                continue
        except (OSError, ValueError):
            pass
        if (failed_root / f"{job['id']}.json").exists():
            print(json.dumps({"event": "skip_failed", "id": job["id"]}), flush=True)
            continue
        argv = list(job.get("argv", []))
        if "--output" in argv:  # the runner supplies the output directory
            index = argv.index("--output")
            del argv[index:index + 2]
        job = {**job, "argv": argv}
        while True:
            used = len(list(attempts_root.glob(f"{job['id']}__a*"))) if attempts_root.is_dir() else 0
            if used > retries:
                failed_root.mkdir(parents=True, exist_ok=True)
                (failed_root / f"{job['id']}.json").write_text(json.dumps({"id": job["id"], "attempts": used}) + "\n")
                failed += 1
                break
            shutil.rmtree(target, ignore_errors=True)
            outcome = core.run({**job, "result_name": name})
            shutil.rmtree(ROOT / "outputs" / name, ignore_errors=True)
            ok = outcome["status"] == "completed" and not outcome.get("error")
            print(json.dumps({"event": "job", "id": job["id"], "attempt": used, "status": outcome["status"],
                              "seconds": round(outcome["job_seconds"], 1), "chunks": len(outcome["chunk_pixel_sha256"]),
                              "peak_gb": round((outcome["peak_allocated_bytes"] or 0) / 1e9, 1),
                              "error": (outcome.get("error") or "")[:400]}), flush=True)
            if ok:
                done += 1
                break
            attempts_root.mkdir(parents=True, exist_ok=True)
            kept = attempts_root / f"{job['id']}__a{used}"
            if target.exists():
                shutil.move(str(target), str(kept))
            else:
                kept.mkdir()
            (kept / "attempt-error.txt").write_text(str(outcome.get("error") or outcome["status"]))
            if q.is_fatal(outcome.get("error")):
                print(json.dumps({"event": "fatal", "id": job["id"]}), flush=True)
                sys.exit(3)
    print(json.dumps({"event": "summary", "completed": done, "failed": failed}), flush=True)


if __name__ == "__main__":
    main()
