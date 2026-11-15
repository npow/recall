"""Persistent native GC research batches, grouped by exact model initialization.

Example: modal run --profile nissan-pow scripts/modal_lingbot_batch.py
  --plan reproduction/lingbot-batch-validation-plan.json
  --output-root reproduction/lingbot-batch-validation --jobs-per-worker 3
"""
import json
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

app = modal.App("worldsync-lingbot-native-batch")


@app.function(image=image, gpu="H100", cpu=8, memory=131072, timeout=3600, max_containers=3,
              volumes={"/cache": weights, "/results": results})
def run_group(group: dict):
    import os
    import shutil
    import subprocess
    import time
    from worldsync.lingbot_batch_rollout import groups

    if len(groups(group["jobs"])) != 1:
        raise ValueError("Launcher must split different initialization configurations")
    weights.reload()
    manifest = json.loads(Path("/work/reproduction/lingbot-weights.json").read_text())
    raw = Path("/cache/14b-causal-fast")
    if any((raw / row["path"]).stat().st_size != row["bytes"] for row in manifest["files"]):
        raise ValueError("Primary checkpoint inventory no longer matches verified staging")
    target = Path("/work/external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast")
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        target.symlink_to(raw)
    identifier = "native-batch-" + str(time.time_ns())
    output = Path("/work/outputs") / identifier
    jobs_path = Path("/tmp") / (identifier + ".json")
    jobs_path.write_text(json.dumps(group))
    destination = Path("/results") / identifier
    published = set()

    def publish():
        source = output / "summary.json"
        if not source.exists():
            return
        summary = json.loads(source.read_text())
        for row in summary["jobs"]:
            folder = output / row["id"]
            if row["id"] not in published:
                destination.mkdir(exist_ok=True)
                if folder.exists():
                    shutil.copytree(folder, destination / row["id"], dirs_exist_ok=True)
                shutil.copy2(source, destination / "summary.json")
                results.commit()
                published.add(row["id"])

    process = subprocess.Popen([sys.executable, "-u", "-m", "worldsync.lingbot_batch_rollout",
                                "--jobs", str(jobs_path), "--output", str(output)], cwd="/work",
                               env=dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                                        WORLDSYNC_LINGBOT_ATTENTION="sdpa"))
    try:
        while process.poll() is None:
            publish()
            time.sleep(2)
        publish()
    finally:
        if output.exists():
            shutil.copytree(output, destination, dirs_exist_ok=True)
            results.commit()
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"status": "failed", "jobs": []}
    manifests = {}
    for job in group["jobs"]:
        path = output / job["id"] / "run.json"
        if path.exists():
            manifests[job["id"]] = json.loads(path.read_text())
    return {"result_directory": identifier, "summary": summary, "runs": manifests,
            "declared_job_ids": [job["id"] for job in group["jobs"]], "configuration": group["configuration"],
            "returncode": process.returncode, "completed_jobs_published": sorted(published)}


@app.local_entrypoint()
def main(plan: str, output_root: str, jobs_per_worker: int = 4):
    from worldsync.lingbot_batch_rollout import groups
    if not 1 <= jobs_per_worker <= 4:
        raise ValueError("Use1–4 serial native jobs per worker")
    specification = json.loads((ROOT / plan).read_text())
    jobs = specification["jobs"]
    for job in jobs:
        if "reference_manifest" in job:
            reference = json.loads((ROOT / job["reference_manifest"]).read_text())["run"]
            if reference["status"] != "completed" or len(reference["chunks"]) != reference["max_chunks"]:
                raise ValueError("Wait for complete fresh references before GPU allocation")
            job["reference"] = reference
    pending = []
    for group in groups(jobs):
        for start in range(0, len(group["jobs"]), jobs_per_worker):
            pending.append({"configuration": group["configuration"], "jobs": group["jobs"][start:start + jobs_per_worker]})
    output = ROOT / output_root
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("group-*.json")) or (output / "launch.json").exists():
        raise ValueError("Preserve existing results; choose an empty output root for a new batch")
    with (output / "launch.json").open("x") as record:
        json.dump({"plan": plan, "job_ids": [job["id"] for job in jobs], "workers": len(pending)}, record, indent=2)
    completed = []
    for index, result in enumerate(run_group.map(pending, return_exceptions=True)):
        if isinstance(result, BaseException):
            result = {"summary": {"status": "failed", "jobs": []}, "error": str(result),
                      "declared_job_ids": [job["id"] for job in pending[index]["jobs"]]}
        for identifier, run in result.pop("runs", {}).items():
            folder = output / identifier
            folder.mkdir(exist_ok=True)
            (folder / "run.json").write_text(json.dumps(run, indent=2) + "\n")
        (output / f"group-{index:03d}.json").write_text(json.dumps(result, indent=2) + "\n")
        completed.append(result)
    returned_ids = [row["id"] for group in completed for row in group["summary"]["jobs"]]
    complete = (sorted(returned_ids) == sorted(job["id"] for job in jobs)
                and all(group["summary"]["status"] == "completed" for group in completed))
    summary = {"scope": specification.get("scope", "persistent native research batch"),
               "declared_jobs": len(jobs), "declared_workers": len(pending),
               "completed_jobs": sum(row["status"] == "completed" for group in completed for row in group["summary"]["jobs"]),
               "status": "completed" if complete else "failed",
               "constructor_calls": sum(group["summary"].get("constructor_calls", 0) for group in completed)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "completed":
        raise RuntimeError("Some batch jobs failed; all returned results and failure records retained")
