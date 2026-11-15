import json

from worldsync import lingbot_episode_queue as q
from worldsync.lingbot_runner_core import job_identifier


def make_job(arm="stock", task="GC007", seed=42):
    return q.episode_job("run1", arm, "gc", task, seed)


def test_episode_job_argv_and_names():
    stock = make_job("stock")
    assert stock["argv"][:6] == ["--split", "gc", "--task-id", "GC007", "--seed", "42"]
    assert "--offload-text-encoder" in stock["argv"] and "--budget-memory" not in stock["argv"]
    budget = make_job("budget")
    assert "--budget-memory" in budget["argv"] and "--offload-text-encoder" not in budget["argv"]
    assert budget["base"] == "adaptive/run1/budget/gc-GC007-seed42"
    assert q.attempt_name(budget, 1) == "adaptive/run1/budget/gc-GC007-seed42__a1"


def test_unknown_arm_rejected():
    try:
        q.episode_job("r", "magic", "gc", "GC007", 1)
    except ValueError as error:
        assert "unknown arm" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_result_name_is_stable_in_runner_core():
    assert job_identifier({"id": "a", "result_name": "adaptive/x/y"}) == "adaptive/x/y"
    assert job_identifier({"id": "a"}).startswith("a-")


def drain(jobs, outcomes, done=(), **kwargs):
    queue = list(jobs)
    records, calls = [], []

    def run_job(job, attempt):
        calls.append((job["id"], attempt))
        return outcomes[(job["id"], attempt)] if (job["id"], attempt) in outcomes else {"status": "completed", "job_seconds": 1}

    summary = q.drain_loop(lambda: queue.pop(0) if queue else None, run_job,
                           lambda job: job["id"] in done, lambda kind, job, payload: records.append((kind, job["id"], payload)),
                           **kwargs)
    return summary, records, calls


def test_resume_skips_completed_episodes():
    jobs = [make_job(task="GC001"), make_job(task="GC002")]
    summary, records, calls = drain(jobs, {}, done={jobs[0]["id"]})
    assert summary["skipped"] == 1 and summary["completed"] == 1
    assert calls == [(jobs[1]["id"], 0)]


def test_transient_failure_retried_then_completes():
    job = make_job()
    outcomes = {(job["id"], 0): {"status": "failed", "error": "PlayerResponseError: invalid player action"}}
    summary, records, calls = drain([job], outcomes)
    assert calls == [(job["id"], 0), (job["id"], 1)]
    assert summary["completed"] == 1 and summary["failed"] == 0
    progress = [payload for kind, _, payload in records if kind == "progress"][0]
    assert [a["status"] for a in progress["attempts"]] == ["failed", "completed"]


def test_attempt_cap_records_failure():
    job = make_job()
    bad = {"status": "failed", "error": "APIStatusError 529 overloaded"}
    outcomes = {(job["id"], k): bad for k in range(5)}
    summary, records, calls = drain([job], outcomes, max_attempts=3)
    assert len(calls) == 3 and summary["failed"] == 1
    assert records[-1][0] == "failed" and len(records[-1][2]["attempts"]) == 3


def test_fatal_cuda_error_aborts_container_and_leaves_queue():
    first, second = make_job(task="GC001"), make_job(task="GC002")
    outcomes = {(first["id"], 0): {"status": "failed", "error": "RuntimeError: CUDA error: illegal memory access"}}
    summary, records, calls = drain([first, second], outcomes)
    assert summary["aborted"] and calls == [(first["id"], 0)]


def test_episode_done_and_latest_attempts():
    job = make_job()
    store = {"adaptive/run1/stock": ["gc-GC007-seed42__a0", "gc-GC008-seed42__a0"],
             "adaptive/run1/stock/gc-GC007-seed42__a0/run.json": {"status": "failed"}}
    def listing(path):
        return store.get(path, [])
    def read(path):
        if path not in store:
            raise OSError(path)
        return store[path]
    assert not q.episode_done(job, listing, read)
    store["adaptive/run1/stock"].append("gc-GC007-seed42__a1")
    store["adaptive/run1/stock/gc-GC007-seed42__a1/run.json"] = {"status": "completed"}
    assert q.episode_done(job, listing, read)
    latest = q.latest_attempts(store["adaptive/run1/stock"])
    assert latest[("gc", "GC007", 42)].endswith("__a1")


def test_throughput_projection():
    rows = [{"status": "completed", "seconds": 400}, {"status": "completed", "seconds": 440}, {"status": "failed", "seconds": 9}]
    out = q.throughput(rows, containers=10, episodes_total=342)
    assert out["episodes"] == 2 and abs(out["mean_episode_seconds"] - 420) < 1e-9
    assert abs(out["projected_hours"] - 342 * 420 / 10 / 3600) < 1e-9
