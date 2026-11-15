"""Predeclared adaptive360 pilot: score stock baselines before memory trials."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import modal

ROOT = Path(__file__).resolve().parents[1]
# Both arms must use the exact source uploaded for the corrected stock run,
# even while collaborators continue changing main.
PILOT_SOURCE_REVISION = "660fa4e93e0b70f4a2df6b7488785d11634ff17e"
sys.path.insert(0, str(ROOT))
if modal.is_local():
    from scripts.modal_lingbot_benchmark import runtime_image, weights, results
    from worldsync.credentials import load_local_credentials
    load_local_credentials()
    credentials = modal.Secret.from_dict({"ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"]})
    frozen_source = tempfile.TemporaryDirectory(prefix="worldsync-adaptive-source-")
    source_path = Path(frozen_source.name)
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only",
                                    PILOT_SOURCE_REVISION, "worldsync"], cwd=ROOT, text=True).splitlines()
    for name in paths:
        target = source_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(["git", "show", f"{PILOT_SOURCE_REVISION}:{name}"], cwd=ROOT))
    image = (runtime_image.pip_install("anthropic==2.8.0")
             .add_local_dir(source_path / "worldsync", "/work/worldsync", ignore=["__pycache__"])
             .add_local_file(ROOT / "reproduction/lingbot-weights.json", "/work/reproduction/lingbot-weights.json")
             .add_local_file(ROOT / "external/PlayWorld/data/gc/data.json", "/work/external/PlayWorld/data/gc/data.json")
             .add_local_dir(ROOT / "external/PlayWorld/data/gc/images", "/work/external/PlayWorld/data/gc/images"))
else:
    image = None
    credentials = modal.Secret.from_dict({})
    weights = modal.Volume.from_name("worldsync-lingbot-weights")
    results = modal.Volume.from_name("worldsync-lingbot-results")

app = modal.App("worldsync-lingbot-adaptive-four-view")


@app.function(image=image, gpu="H200", cpu=8, memory=131072, timeout=1800, max_containers=2,
              secrets=[credentials], volumes={"/cache": weights, "/results": results})
def generate(seed: int, memory: bool):
    import shutil
    import subprocess
    import time
    weights.reload()
    target = Path("/work/external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast")
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        target.symlink_to("/cache/14b-causal-fast")
    mode = "memory" if memory else "stock"
    identifier = f"adaptive-four-view-{mode}-GC027-seed{seed}-{time.time_ns()}"
    output = Path("/work/outputs") / identifier
    command = [sys.executable, "-u", "-m", "worldsync.lingbot_rollout", "--split", "gc", "--task-id", "GC027",
               "--seed", str(seed), "--max-chunks", "40", "--turn-degrees-per-second", "30",
               "--adaptive-retrieval-pilot", "--offload-text-encoder", "--output", str(output)]
    if memory:
        command.append("--pose-retrieval")
    try:
        process = subprocess.run(command, cwd="/work", env={**os.environ,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "WORLDSYNC_LINGBOT_ATTENTION": "sdpa"})
    finally:
        if output.exists():
            shutil.copytree(output, Path("/results") / identifier)
            results.commit()
    run = json.loads((output / "run.json").read_text()) if (output / "run.json").exists() else None
    return {"mode": mode, "seed": seed, "workspace": "nissan-pow", "result_directory": identifier,
            "returncode": process.returncode, "run": run,
            "scope": "Local adaptive pilot, not published AgentPlayer score reproduction"}


@app.local_entrypoint()
def main(memory: bool = False):
    plan_path = ROOT / "reproduction/adaptive-four-view-plan.json"
    plan = json.loads(plan_path.read_text())
    case = plan["case"]
    task = next(row for row in json.loads((ROOT / "external/PlayWorld/data/gc/data.json").read_text()) if row["task_id"] == case["task_id"])
    if (hashlib.sha256((ROOT / "external/PlayWorld/data" / task["image_path"]).read_bytes()).hexdigest() != case["image_sha256"]
            or hashlib.sha256(task["image_caption"].encode()).hexdigest() != case["caption_sha256"]):
        raise ValueError("Audited source input changed")
    if memory:
        scores_path = ROOT / "reproduction/adaptive-four-view/stock-summary.json"
        scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
        if (not scores.get("complete") or scores.get("frozen_plan_sha256") != hashlib.sha256(plan_path.read_bytes()).hexdigest()
                or {row["seed"] for row in scores["results"]} != set(plan["seeds"])):
            raise ValueError("Score both stock baselines before launching the memory candidate")
    mode = "memory" if memory else "stock"
    jobs = []
    for seed in plan["seeds"]:
        path = ROOT / "outputs/lingbot-reproduction" / f"adaptive-four-view-{mode}-seed{seed}.json"
        if path.exists():
            print(f"Retaining completed or failed attempt for{mode}/seed{seed}; do not overwrite", flush=True)
        else:
            jobs.append((seed, memory))
    failures = []
    for index, result in enumerate(generate.starmap(jobs, return_exceptions=True)):
        if isinstance(result, BaseException):
            failures.append({"job": jobs[index], "error": str(result)})
            continue
        result["frozen_plan_sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        result["source_revision"] = PILOT_SOURCE_REVISION
        path = ROOT / "outputs/lingbot-reproduction" / f"adaptive-four-view-{mode}-seed{result['seed']}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "run"}, indent=2), flush=True)
    if failures:
        path = ROOT / "outputs/lingbot-reproduction" / f"adaptive-four-view-{mode}-failures.json"
        path.write_text(json.dumps(failures, indent=2) + "\n")
        raise RuntimeError("Some attempts failed; retain all failures before retrying")
