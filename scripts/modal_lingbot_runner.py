"""Persistent multi-run LingBot-World2 runner on Modal (nissan-pow).

The 14B pipeline loads once per container (100-155 s); every job after that reuses it.
Warm containers stay up for LINGBOT_RUNNER_SCALEDOWN seconds, so a deployed app also serves
later client sessions without reloading.

    modal deploy scripts/modal_lingbot_runner.py              # keep it around (min_containers=0)
    python scripts/lingbot_runner_client.py jobs.jsonl --gpu H100 --containers 2

Or run it ephemerally: `modal run scripts/modal_lingbot_runner.py --jobs jobs.jsonl --gpu H100`.
`--mode oneshot` runs each job in a fresh process (full reload) for timing and bit-exactness checks.
Job format: see worldsync/lingbot_runner_core.py.

GPU choice: H200 for memory-heavy methods (46/6, budget memory, KV retention, DeCoPrune);
H100 80 GB for stock-sized runs. H100 jobs default to lossless savings: expandable-segments
allocator and CPU offload of the text encoder after encoding. No quantization.
"""
import json
import os
from pathlib import Path
import sys
import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if modal.is_local():
    from scripts.modal_lingbot_benchmark import image, weights, results
else:
    image = None
    weights = modal.Volume.from_name("worldsync-lingbot-weights")
    results = modal.Volume.from_name("worldsync-lingbot-results")

APP_NAME = "worldsync-lingbot-runner"
app = modal.App(APP_NAME)
MAX_CONTAINERS = int(os.environ.get("LINGBOT_RUNNER_MAX_CONTAINERS", "2"))
SCALEDOWN = int(os.environ.get("LINGBOT_RUNNER_SCALEDOWN", "600"))
VOLUMES = {"/cache": weights, "/results": results}
H100_DEFAULTS = {"offload_text_encoder": True}
WARMUP_JOB = {"id": "warmup", "actions": ["WAIT"], "task_id": "GC007"}


def _prepare_container(expandable):
    if expandable:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    weights.reload()
    target = Path("/work/external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to("/cache/14b-causal-fast")


def _start_core(defaults, expandable):
    _prepare_container(expandable)
    from worldsync.lingbot_runner_core import RunnerCore
    def before_job(job):
        if "--lora" in list(job.get("extra", [])) + list(job.get("argv", [])):
            results.reload()  # adapters written by other containers become visible
    core = RunnerCore(commit=results.commit, defaults=defaults, before_job=before_job)
    try:  # Load the model and pay CUDA's first-call tax at container start, not in the first job.
        warm = core.run({**WARMUP_JOB}, persist=False)
        print(json.dumps({"event": "warmup", "status": warm["status"], "error": warm["error"],
                          "seconds": warm["job_seconds"]}), flush=True)
    except BaseException as failure:
        print(f"warmup failed, first job will load instead: {failure}", flush=True)
    return core


@app.cls(image=image, gpu="H200", cpu=8, memory=196608, timeout=3600, startup_timeout=1800,
         scaledown_window=SCALEDOWN, max_containers=MAX_CONTAINERS, volumes=VOLUMES)
class RunnerH200:
    @modal.enter()
    def start(self):
        self.core = _start_core({}, expandable=False)

    @modal.method()
    def run(self, job: dict):
        return self.core.run(job)


@app.cls(image=image, gpu="H100", cpu=8, memory=131072, timeout=3600, startup_timeout=1800,
         scaledown_window=SCALEDOWN, max_containers=MAX_CONTAINERS, volumes=VOLUMES)
class RunnerH100:
    @modal.enter()
    def start(self):
        self.core = _start_core(H100_DEFAULTS, expandable=True)

    @modal.method()
    def run(self, job: dict):
        return self.core.run(job)


def _oneshot(job, defaults):
    weights.reload()
    if "--lora" in list(job.get("extra", [])) + list(job.get("argv", [])):
        results.reload()
    _prepare_container(expandable=False)
    from worldsync.lingbot_runner_core import run_oneshot
    return run_oneshot({**defaults, **job}, commit=results.commit)


@app.function(image=image, gpu="H200", cpu=8, memory=196608, timeout=3600, max_containers=MAX_CONTAINERS, volumes=VOLUMES)
def oneshot_h200(job: dict):
    return _oneshot(job, {})


@app.function(image=image, gpu="H100", cpu=8, memory=131072, timeout=3600, max_containers=MAX_CONTAINERS, volumes=VOLUMES)
def oneshot_h100(job: dict):
    return _oneshot(job, {**H100_DEFAULTS, "allocator_expandable": True})


@app.local_entrypoint()
def main(jobs: str, gpu: str = "H200", mode: str = "warm", containers: int = 2, out: str = ""):
    """Run a jobs.jsonl through this (ephemeral) app; see scripts/lingbot_runner_client.py for deployed use."""
    from concurrent.futures import ThreadPoolExecutor
    import time
    if gpu not in {"H100", "H200"} or mode not in {"warm", "oneshot"}:
        raise ValueError("gpu is H100|H200 and mode is warm|oneshot")
    from scripts.lingbot_runner_client import expand
    entries = [expand(json.loads(line)) for line in Path(jobs).read_text().splitlines() if line.strip()]
    if mode == "warm":
        call = (RunnerH100 if gpu == "H100" else RunnerH200)().run.remote
    else:
        call = (oneshot_h100 if gpu == "H100" else oneshot_h200).remote
    started = time.monotonic()
    def one(job):
        sent = time.monotonic()
        result = call(job)
        result["client_wall_s"] = time.monotonic() - sent
        return result
    with ThreadPoolExecutor(max_workers=containers) as pool:
        outcomes = list(pool.map(one, entries))
    total = time.monotonic() - started
    destination = Path(out) if out else ROOT / "outputs/lingbot-reproduction" / f"runner-{mode}-{gpu}-{int(time.time())}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"mode": mode, "gpu": gpu, "containers": containers, "total_wall_s": total,
                                       "results": outcomes}, indent=1) + "\n")
    for outcome in outcomes:
        print(json.dumps({key: outcome.get(key) for key in ("id", "status", "error", "job_seconds", "client_wall_s", "result_directory")}))
    print(f"wrote {destination} ({total:.0f} s total for {len(entries)} jobs)")
