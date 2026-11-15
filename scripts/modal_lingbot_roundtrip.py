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


def mode_name(retrieval, cache_frames, rephase_sink, positions, retrieved_frames=4, selection="nearest"):
    if rephase_sink:
        return "sink-rephase"
    if not retrieval:
        return "stock" if cache_frames == 18 else f"context{cache_frames}"
    base = "retrieval-native" if positions == "native" else "retrieval"
    return base if retrieved_frames == 4 and selection == "nearest" else f"{base}-{retrieved_frames}-{selection}"


@app.function(image=image, gpu="H200", cpu=8, memory=131072, timeout=1800, max_containers=3,
              volumes={"/cache": weights, "/results": results})
def generate(path_name: str = "round_trip", seed: int = 42, retrieval: bool = False, cache_frames: int = 18, task_id: str = "GC007", rephase_sink: bool = False, retrieval_positions: str = "virtual", turn_rate: float = 60, offload_text_encoder: bool = False, retrieved_frames: int = 4, retrieval_selection: str = "nearest"):
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
    mode = mode_name(retrieval, cache_frames, rephase_sink, retrieval_positions, retrieved_frames, retrieval_selection)
    identifier = f"{mode}-{task_id}-{path_name}-seed{seed}-{time.time_ns()}"
    output = Path("/work/outputs") / identifier
    try:
        command = [sys.executable, "-u", "-m", "worldsync.lingbot_rollout", "--split", "gc",
            "--task-id", task_id, "--seed", str(seed), "--turn-degrees-per-second", str(turn_rate), "--max-chunks", str(len(actions)),
            "--native-fixed-controls", "--fixed-actions", *actions, "--output", str(output)]
        if offload_text_encoder:
            command.append("--offload-text-encoder")
        if retrieval:
            command.extend(["--pose-retrieval", "--retrieval-positions", retrieval_positions,
                            "--retrieved-frames", str(retrieved_frames), "--retrieval-selection", retrieval_selection])
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


@app.function(image=image, gpu="H100", cpu=8, memory=131072, timeout=1800, max_containers=3,
              volumes={"/cache": weights, "/results": results})
def generate_h100(path_name: str, seed: int, retrieval: bool, cache_frames: int, task_id: str,
                  rephase_sink: bool, retrieval_positions: str, turn_rate: float,
                  retrieved_frames: int = 4, retrieval_selection: str = "nearest"):
    # Same GPU encoding and denoising arithmetic; only inactive encoder weights
    # move to CPU after the prompt context is computed.
    return generate.local(path_name, seed, retrieval, cache_frames, task_id,
                          rephase_sink, retrieval_positions, turn_rate, True, retrieved_frames, retrieval_selection)


@app.local_entrypoint()
def main(path_name: str = "round_trip", seed: int = 42, retrieval: bool = False, cache_frames: int = 18, task_id: str = "GC007", validation: bool = False, memory_validation: bool = False, task_benchmark: bool = False, rephase_sink: bool = False, retrieval_positions: str = "virtual", turn_rate: float = 60, retrieved_frames: int = 4, retrieval_selection: str = "nearest", multi_view_ablation: bool = False):
    if sum([validation, memory_validation, task_benchmark, multi_view_ablation]) > 1:
        raise ValueError("Choose one frozen experiment plan")
    if retrieved_frames not in {4, 8} or retrieval_selection not in {"nearest", "diverse"}:
        raise ValueError("retrieval supports4/8 frames and nearest/diverse selection")
    if (retrieved_frames != 4 or retrieval_selection != "nearest") and not retrieval:
        raise ValueError("non-default memory settings require retrieval mode")
    if retrieval_positions not in {"native", "virtual"} or (retrieval_positions != "virtual" and not retrieval):
        raise ValueError("native retrieval positions require retrieval mode")
    if cache_frames not in {18, 34} or ((retrieval or rephase_sink) and cache_frames != 18) or (retrieval and rephase_sink):
        raise ValueError("This declared pilot supports stock18, context34, or stock-budget retrieval")
    if multi_view_ablation:
        plan = json.loads((ROOT / "reproduction/multiview-retrieval-plan.json").read_text())
        case = plan["case"]
        jobs = [(case["path"], case["seed"], True, 18, case["task_id"], False, "virtual", 60,
                 variant["retrieved_frames"], variant["selection"]) for variant in plan["variants"]]
        failures = []
        for index, result in enumerate(generate_h100.starmap(jobs, return_exceptions=True)):
            if isinstance(result, BaseException):
                failures.append({"job": jobs[index], "error": str(result)})
                (ROOT / "outputs/lingbot-reproduction/multiview-generation-failures.json").write_text(json.dumps(failures, indent=2) + "\n")
                continue
            output = ROOT / "outputs/lingbot-reproduction" / f"multiview-{case['task_id']}-{result['mode']}-{case['path']}-seed{case['seed']}.json"
            output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({k: v for k, v in result.items() if k != "run"}, indent=2), flush=True)
        return
    if task_benchmark:
        if validation or memory_validation:
            raise ValueError("Choose one experiment plan")
        plan = json.loads((ROOT / "reproduction/task-memory-benchmark-plan.json").read_text())
        jobs = [(case["path"], case["seed"], retrieve, 18, case["task_id"], False, "virtual", plan["turn_degrees_per_second"])
                for case in plan["cases"] for retrieve in (False, True)]
        for index, result in enumerate(generate_h100.starmap(jobs, return_exceptions=True)):
            job = jobs[index]
            if isinstance(result, BaseException):
                print(f"FAILED task benchmark attempt {job}: {result}", flush=True)
                continue
            run = result["run"]
            output = ROOT / "outputs/lingbot-reproduction" / f"task-benchmark-{run['task_id']}-{result['mode']}-{result['path']}-seed{run['seed']}.json"
            output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({k: v for k, v in result.items() if k != "run"}, indent=2), flush=True)
        return
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
        for index, result in enumerate(generate.starmap(pending, return_exceptions=True)):
            job = pending[index]
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
    result = generate.remote(path_name, seed, retrieval, cache_frames, task_id, rephase_sink, retrieval_positions, turn_rate, False, retrieved_frames, retrieval_selection)
    mode = mode_name(retrieval, cache_frames, rephase_sink, retrieval_positions, retrieved_frames, retrieval_selection)
    scene = "" if task_id == "GC007" else f"{task_id}-"
    output = ROOT / "outputs/lingbot-reproduction" / f"native-{scene}{mode}-{path_name}-seed{seed}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "run"}, indent=2))
