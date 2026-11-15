"""Stock native-control round-trip calibration; no adaptive player or memory edit."""
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

app = modal.App("worldsync-lingbot-roundtrip")


@app.function(image=image, gpu="H200", cpu=8, memory=131072, timeout=1800, max_containers=3,
              volumes={"/cache": weights, "/results": results})
def generate(path_name: str = "round_trip", seed: int = 42, retrieval: bool = False, cache_frames: int = 18, task_id: str = "GC007", rephase_sink: bool = False, retrieval_positions: str = "virtual"):
    import shutil
    import subprocess
    import time
    from worldsync.lingbot_paths import PATHS
    actions = PATHS[path_name]
    weights.reload()
    target = Path("/work/external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast")
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        target.symlink_to("/cache/14b-causal-fast")
    mode = "sink-rephase" if rephase_sink else (("retrieval-native" if retrieval_positions == "native" else "retrieval") if retrieval else ("stock" if cache_frames == 18 else f"context{cache_frames}"))
    identifier = f"{mode}-{task_id}-{path_name}-seed{seed}-{time.time_ns()}"
    output = Path("/work/outputs") / identifier
    try:
        command = [sys.executable, "-u", "-m", "worldsync.lingbot_rollout", "--split", "gc",
            "--task-id", task_id, "--seed", str(seed), "--max-chunks", str(len(actions)),
            "--native-fixed-controls", "--fixed-actions", *actions, "--output", str(output)]
        if retrieval:
            command.extend(["--pose-retrieval", "--retrieval-positions", retrieval_positions])
        if rephase_sink:
            command.append("--rephase-sink")
        if cache_frames != 18:
            command.extend(["--controlled-ablation", "--local-attn-size", str(cache_frames)])
        outcome = subprocess.run(command, cwd="/work")
    finally:
        if output.exists():
            shutil.copytree(output, Path("/results") / identifier)
            results.commit()
    if outcome.returncode:
        raise RuntimeError(f"Calibration failed; artifacts preserved: {identifier}")
    report = json.loads((output / "run.json").read_text())
    return {"scope": "fixed-path native-camera controlled experiment; not PlayWorld task score", "mode": mode,
            "path": path_name, "result_directory": identifier, "run": report}


@app.local_entrypoint()
def main(path_name: str = "round_trip", seed: int = 42, retrieval: bool = False, cache_frames: int = 18, task_id: str = "GC007", validation: bool = False, memory_validation: bool = False, rephase_sink: bool = False, retrieval_positions: str = "virtual"):
    if retrieval_positions not in {"native", "virtual"} or (retrieval_positions != "virtual" and not retrieval):
        raise ValueError("native retrieval positions require retrieval mode")
    if cache_frames not in {18, 34} or ((retrieval or rephase_sink) and cache_frames != 18) or (retrieval and rephase_sink):
        raise ValueError("This declared pilot supports stock18, context34, or stock-budget retrieval")
    if memory_validation:
        if validation:
            raise ValueError("Choose one frozen validation plan")
        plan = json.loads((ROOT / "reproduction/memory-validation-plan.json").read_text())
        jobs = [(case["path"], case["seed"], retrieval, 18, case["task_id"], False, "virtual")
                for case in plan["cases"] for retrieval in (False, True)]
        pending = []
        for job in jobs:
            path, seed, retrieve, _, task, _, _ = job
            mode = "retrieval" if retrieve else "stock"
            output = ROOT / "outputs/lingbot-reproduction" / f"memory-validation-{task}-{mode}-{path}-seed{seed}.json"
            if output.exists() and json.loads(output.read_text())["run"]["status"] == "completed":
                print(f"Reusing completed {task}/{path}/{mode}/seed{seed}", flush=True)
            else:
                pending.append(job)
        failures_path = ROOT / "outputs/lingbot-reproduction/memory-validation-attempt-failures.json"
        failures = json.loads(failures_path.read_text()) if failures_path.exists() else []
        for job, result in zip(pending, generate.starmap(pending, return_exceptions=True)):
            if isinstance(result, BaseException):
                failures.append({"job": job, "error": str(result)})
                failures_path.write_text(json.dumps(failures, indent=2) + "\n")
                print(f"Preserved failed attempt: {job}: {result}", flush=True)
                continue
            run = result["run"]
            output = ROOT / "outputs/lingbot-reproduction" / f"memory-validation-{run['task_id']}-{result['mode']}-{result['path']}-seed{run['seed']}.json"
            output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({k: v for k, v in result.items() if k != "run"}, indent=2), flush=True)
        return
    if validation:
        plan = json.loads((ROOT / "reproduction/context-validation-plan.json").read_text())
        jobs = [(plan["path"], case["seed"], False, cache, case["task_id"])
                for case in plan["cases"] for cache in (18, 34)]
        for result in generate.starmap(jobs):
            run = result["run"]
            output = ROOT / "outputs/lingbot-reproduction" / f"validation-{run['task_id']}-{result['mode']}-seed{run['seed']}.json"
            output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({k: v for k, v in result.items() if k != "run"}, indent=2), flush=True)
        return
    result = generate.remote(path_name, seed, retrieval, cache_frames, task_id, rephase_sink, retrieval_positions)
    mode = "sink-rephase" if rephase_sink else (("retrieval-native" if retrieval_positions == "native" else "retrieval") if retrieval else ("stock" if cache_frames == 18 else f"context{cache_frames}"))
    scene = "" if task_id == "GC007" else f"{task_id}-"
    output = ROOT / "outputs/lingbot-reproduction" / f"native-{scene}{mode}-{path_name}-seed{seed}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "run"}, indent=2))
