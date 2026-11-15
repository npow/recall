"""Queue logic for the warm adaptive-episode runner (pure Python, unit-testable without Modal).

A drain worker owns one warm LingBot container. It pops episode jobs from a shared queue,
skips episodes whose result is already in the results volume (resume), runs each through the
warm `RunnerCore`, retries transient failures (player/API hiccups) up to a cap, and writes a
small progress record per episode so throughput can be read off the volume.

Result layout (all under the results volume):

    adaptive/<label>/<arm>/<split>-<TASK>-seed<N>__a<k>/   one directory per attempt (rollout.mp4, run.json, ...)
    adaptive/<label>/progress/<arm>-<split>-<TASK>-seed<N>.json
    adaptive/<label>/failed/<arm>-<split>-<TASK>-seed<N>.json
"""
import json
import re
import time

# Memory arms the parent may freeze later; extra rollout flags are appended after the common ones.
ARMS = {
    "stock": {"gpu": "H100", "extra": []},
    "budget": {"gpu": "H200", "extra": ["--budget-memory", "--budget-retrieved", "10", "--budget-rope", "native"]},
}
COMMON = ["--max-chunks", "40"]
H100_SAVINGS = ["--offload-text-encoder"]  # lossless; stock peaks ~80 GB otherwise
TRANSIENT = ("PlayerResponseError", "invalid player", "overloaded", "529", "rate limit", "RateLimit", "timed out",
             "timeout", "APIConnectionError", "APIStatusError", "InternalServerError", "ConnectionError",
             "Connection reset", "503", "502", "500")
FATAL = ("CUDA error", "illegal memory access", "device-side assert", "CUDA out of memory", "OutOfMemoryError")


def case_key(arm, split, task_id, seed):
    return f"{arm}-{split}-{task_id}-seed{seed}"


def base_name(label, arm, split, task_id, seed):
    return f"adaptive/{label}/{arm}/{split}-{task_id}-seed{seed}"


def episode_job(label, arm, split, task_id, seed, extra=None):
    """The warm-runner job (raw rollout argv) for one adaptive, player-driven episode."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}; declared arms: {sorted(ARMS)}")
    argv = ["--split", split, "--task-id", task_id, "--seed", str(seed), *COMMON]
    if ARMS[arm]["gpu"] == "H100":
        argv += H100_SAVINGS
    argv += list(ARMS[arm]["extra"]) + list(extra or [])
    return {"id": case_key(arm, split, task_id, seed), "arm": arm, "split": split, "task_id": task_id,
            "seed": seed, "label": label, "base": base_name(label, arm, split, task_id, seed), "argv": argv}


def attempt_name(job, attempt):
    return f"{job['base']}__a{attempt}"


def is_transient(error):
    text = str(error or "")
    return any(token in text for token in TRANSIENT) and not any(token in text for token in FATAL)


def is_fatal(error):
    return any(token in str(error or "") for token in FATAL)


def completed(result):
    return bool(result) and result.get("status") == "completed" and not result.get("error")


def episode_done(job, list_directory, read_json):
    """True when some attempt directory of this episode holds a completed run.json.

    list_directory(path) -> names under path (empty list if missing); read_json(path) -> dict.
    """
    parent, _, stem = job["base"].rpartition("/")
    for name in list_directory(parent):
        if name.startswith(stem + "__a"):
            try:
                if read_json(f"{parent}/{name}/run.json").get("status") == "completed":
                    return True
            except (OSError, ValueError):
                continue
    return False


def drain_loop(next_job, run_job, is_done, record, max_attempts=3, max_episodes=None, now=time.time):
    """Pop jobs until the queue is empty. Returns a summary dict.

    next_job():            job dict or None when the queue is empty
    run_job(job, attempt): runs one attempt, returns the RunnerCore result dict
    is_done(job):          True if a completed attempt already exists (resume)
    record(kind, job, payload): persists progress/failed records
    """
    summary = {"completed": 0, "skipped": 0, "failed": 0, "attempts": 0, "episode_seconds": [], "aborted": None}
    while max_episodes is None or summary["completed"] + summary["failed"] < max_episodes:
        job = next_job()
        if job is None:
            break
        if is_done(job):
            summary["skipped"] += 1
            record("progress", job, {"status": "skipped_already_complete", "finished_at": now()})
            continue
        started = now()
        attempts = []
        result = None
        for attempt in range(max_attempts):
            summary["attempts"] += 1
            result = run_job(job, attempt)
            attempts.append({"attempt": attempt, "status": result.get("status"), "error": (result.get("error") or "")[:400],
                             "job_seconds": result.get("job_seconds")})
            if completed(result):
                break
            if is_fatal(result.get("error")):
                summary["aborted"] = "fatal error; container is no longer trustworthy"
                break
            if not is_transient(result.get("error")) and result.get("status") not in ("failed", "empty_rollout", "unknown"):
                break
        seconds = now() - started
        if completed(result):
            summary["completed"] += 1
            summary["episode_seconds"].append(seconds)
            run = result.get("run") or {}
            record("progress", job, {"status": "completed", "seconds": seconds, "attempts": attempts,
                                      "result_directory": result.get("result_directory"), "finished_at": now(),
                                      "generated_seconds": run.get("generated_seconds"), "termination": run.get("termination"),
                                      "peak_allocated_bytes": result.get("peak_allocated_bytes"),
                                      "weights_unchanged": result.get("weights_unchanged")})
        else:
            summary["failed"] += 1
            record("failed", job, {"status": "failed", "seconds": seconds, "attempts": attempts, "finished_at": now()})
        if summary["aborted"]:
            break
    return summary


def throughput(progress_records, containers, episodes_total=342):
    """Projection from measured per-episode seconds (episode wall time inside a warm container)."""
    seconds = [row["seconds"] for row in progress_records if row.get("status") == "completed" and row.get("seconds")]
    if not seconds:
        return {}
    mean = sum(seconds) / len(seconds)
    return {"episodes": len(seconds), "mean_episode_seconds": mean, "episodes_per_hour_per_container": 3600 / mean,
            "projected_hours": episodes_total * mean / containers / 3600, "containers": containers,
            "episodes_total": episodes_total}


def name_pattern():
    return re.compile(r"^(?P<split>gc|if|insight|outsight)-(?P<task>[A-Z]+\d+)-seed(?P<seed>\d+)__a(?P<attempt>\d+)$")


def latest_attempts(names):
    """{(split, task, seed): highest attempt directory name} from a list of directory names."""
    pattern = name_pattern()
    best = {}
    for name in names:
        match = pattern.match(name)
        if not match:
            continue
        key = (match["split"], match["task"], int(match["seed"]))
        if key not in best or int(match["attempt"]) > best[key][0]:
            best[key] = (int(match["attempt"]), name)
    return {key: value[1] for key, value in best.items()}
