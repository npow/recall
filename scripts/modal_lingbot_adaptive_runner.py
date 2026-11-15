"""Warm adaptive-episode runner for LingBot-World2 on Modal (nissan-pow).

Containers load the 14B pipeline once (~110 s), then pull ADAPTIVE player-driven episodes from a
modal.Queue until it is empty, skipping episodes already in the results volume and retrying
transient failures. One config per arm: stock 18/6 on H100-style memory settings, budget memory
on H200. See reproduction/fast-full-run.md and scripts/adaptive_runner_client.py.

    MODAL_PROFILE=nissan-pow ADAPTIVE_MAX_CONTAINERS=5 modal deploy scripts/modal_lingbot_adaptive_runner.py
    python scripts/adaptive_runner_client.py submit --label full1 --arms stock,budget --all --containers stock=5,budget=5
"""
import json
import os
from pathlib import Path
import shutil
import sys

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
APP_NAME = "worldsync-lingbot-adaptive-runner"
MAX_CONTAINERS = int(os.environ.get("ADAPTIVE_MAX_CONTAINERS", "5"))
SCALEDOWN = int(os.environ.get("ADAPTIVE_SCALEDOWN", "300"))
if modal.is_local():
    from scripts.modal_lingbot_benchmark import runtime_image, weights, results
    from worldsync.credentials import load_local_credentials
    load_local_credentials()
    credentials = modal.Secret.from_dict({"ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"]})
    data = ROOT / "external/PlayWorld/data"
    image = (runtime_image.pip_install("anthropic==2.8.0")
             .add_local_dir(ROOT / "worldsync", "/work/worldsync", ignore=["__pycache__"])
             .add_local_file(ROOT / "reproduction/lingbot-weights.json", "/work/reproduction/lingbot-weights.json")
             .add_local_file(ROOT / "reproduction/state.json", "/work/reproduction/state.json"))
    for split in ("gc", "if", "insight", "outsight"):
        image = (image.add_local_file(data / split / "data.json", f"/work/external/PlayWorld/data/{split}/data.json")
                 .add_local_dir(data / split / "images", f"/work/external/PlayWorld/data/{split}/images"))
else:
    image = None
    credentials = modal.Secret.from_dict({})
    weights = modal.Volume.from_name("worldsync-lingbot-weights")
    results = modal.Volume.from_name("worldsync-lingbot-results")

app = modal.App(APP_NAME)
VOLUMES = {"/cache": weights, "/results": results}
WARMUP_JOB = {"id": "warmup", "actions": ["WAIT"], "task_id": "GC007"}
CANARY_JOB = {"id": "canary", "path": "turn_return", "task_id": "GC007", "seed": 42, "offload_text_encoder": True}


def _start_core(expandable):
    if expandable:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("WORLDSYNC_LINGBOT_ATTENTION", "sdpa")
    weights.reload()
    target = Path("/work/external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to("/cache/14b-causal-fast")
    from worldsync.lingbot_runner_core import RunnerCore
    core = RunnerCore(commit=results.commit)
    try:  # load the model and pay CUDA's first-call tax at container start, not in the first episode
        warm = core.run({**WARMUP_JOB, "offload_text_encoder": expandable}, persist=False)
        print(json.dumps({"event": "warmup", "status": warm["status"], "seconds": warm["job_seconds"]}), flush=True)
    except BaseException as failure:
        print(f"warmup failed, first episode will load instead: {failure}", flush=True)
    return core


def _list_directory(path):
    try:
        return sorted(os.listdir(Path("/results") / path))
    except OSError:
        return []


def _read_json(path):
    return json.loads((Path("/results") / path).read_text())


def _drain(core, queue_name, label, max_episodes, max_attempts, canary):
    import time
    from worldsync import lingbot_episode_queue as q
    queue = modal.Queue.from_name(queue_name, create_if_missing=True)

    def next_job():
        return queue.get(block=False)

    def is_done(job):
        results.reload()
        return q.episode_done(job, _list_directory, _read_json)

    def run_job(job, attempt):
        name = q.attempt_name(job, attempt)
        outcome = core.run({**job, "result_name": name})
        shutil.rmtree(Path("/work/outputs") / name, ignore_errors=True)  # keep the container disk small
        outcome.pop("run", None)
        return outcome

    def record(kind, job, payload):
        directory = Path("/results/adaptive") / label / kind
        directory.mkdir(parents=True, exist_ok=True)
        payload = {**payload, "arm": job["arm"], "split": job["split"], "task_id": job["task_id"], "seed": job["seed"],
                   "base": job["base"], "container_pid": os.getpid()}
        (directory / f"{job['id']}.json").write_text(json.dumps(payload, indent=1) + "\n")
        results.commit()

    canary_hashes = []
    if canary:
        first = core.run({**CANARY_JOB, "result_name": f"adaptive/{label}/canary/{time.time_ns()}-before"})
        canary_hashes.append(first["chunk_pixel_sha256"])
    summary = q.drain_loop(next_job, run_job, is_done, record, max_attempts=max_attempts, max_episodes=max_episodes)
    if canary:
        last = core.run({**CANARY_JOB, "result_name": f"adaptive/{label}/canary/{time.time_ns()}-after"})
        canary_hashes.append(last["chunk_pixel_sha256"])
        summary["canary_identical_after_episodes"] = (canary_hashes[0] == canary_hashes[1] and bool(canary_hashes[0]))
        summary["canary_weights_unchanged"] = last["weights_unchanged"]
    summary["warm_jobs_run"] = core.runtime.jobs_run
    return summary


@app.cls(image=image, gpu="H200", cpu=8, memory=196608, timeout=4 * 3600, startup_timeout=1800,
         scaledown_window=SCALEDOWN, max_containers=MAX_CONTAINERS, secrets=[credentials], volumes=VOLUMES)
class AdaptiveH200:
    @modal.enter()
    def start(self):
        self.core = _start_core(expandable=False)

    @modal.method()
    def drain(self, queue_name: str, label: str, max_episodes: int | None = None, max_attempts: int = 3,
              canary: bool = False):
        return _drain(self.core, queue_name, label, max_episodes, max_attempts, canary)


@app.cls(image=image, gpu="H100", cpu=8, memory=131072, timeout=4 * 3600, startup_timeout=1800,
         scaledown_window=SCALEDOWN, max_containers=MAX_CONTAINERS, secrets=[credentials], volumes=VOLUMES)
class AdaptiveH100:
    @modal.enter()
    def start(self):
        self.core = _start_core(expandable=True)

    @modal.method()
    def drain(self, queue_name: str, label: str, max_episodes: int | None = None, max_attempts: int = 3,
              canary: bool = False):
        return _drain(self.core, queue_name, label, max_episodes, max_attempts, canary)


@app.function(image=image, cpu=2, memory=4096, timeout=600, volumes={"/results": results})
def status(label: str):
    """Counts, per-arm progress records and the throughput projection, read from the results volume."""
    from worldsync import lingbot_episode_queue as q
    results.reload()
    out = {"label": label, "arms": {}}
    root = Path("/results/adaptive") / label
    for kind in ("progress", "failed"):
        for path in sorted((root / kind).glob("*.json")) if (root / kind).is_dir() else []:
            row = json.loads(path.read_text())
            arm = out["arms"].setdefault(row["arm"], {"completed": [], "failed": [], "skipped": 0})
            if kind == "failed":
                arm["failed"].append({key: row.get(key) for key in ("split", "task_id", "seed", "attempts")})
            elif row["status"] == "completed":
                arm["completed"].append(row)
            else:
                arm["skipped"] += 1
    for arm, rows in out["arms"].items():
        rows["completed_count"] = len(rows["completed"])
        rows["throughput"] = q.throughput(rows["completed"], containers=1)
    return out


@app.local_entrypoint()
def main(label: str, arm: str = "stock", split: str = "gc", task_id: str = "GC007", seed: int = 42):
    """Smallest end-to-end check: one episode through a single warm container."""
    from worldsync import lingbot_episode_queue as q
    queue = modal.Queue.from_name(f"{label}-{arm}", create_if_missing=True)
    queue.put(q.episode_job(label, arm, split, task_id, seed))
    cls = AdaptiveH200 if q.ARMS[arm]["gpu"] == "H200" else AdaptiveH100
    print(json.dumps(cls().drain.remote(f"{label}-{arm}", label, 1, 3, False), indent=1))
