"""Four-dimension (GC / IF / insight / out-of-sight) stock vs budget-memory adaptive episodes on Modal.

48 cases (12 per dimension, salted-hash selection in worldsync/fourdim.py) x arms. Each episode runs the
released-style Haiku reference-guided player (insight cases hold WAIT with no player), with
--player-retries 2 and --text-prompt both on every arm; the arms differ ONLY in the memory setting.
Opus (worldsync.judge) scores inside Modal; Gemini scoring runs locally from downloaded videos
(scripts/build_fourdim_manifest.py + worldsync.gemini_vertex_judge).

  MODAL_PROFILE=nissan-pow modal run scripts/modal_lingbot_fourdim.py --seed 42
"""
import json
from pathlib import Path
import sys
import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if modal.is_local():
    import os
    from scripts.modal_lingbot_benchmark import runtime_image, weights, results
    from worldsync.credentials import load_local_credentials
    load_local_credentials()
    credentials = modal.Secret.from_dict({"ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"]})
    image = (runtime_image.pip_install("anthropic==2.8.0")
        .run_commands("git clone https://github.com/hku-sail/PlayWorld.git /work/external/PlayWorld",
                      "git -C /work/external/PlayWorld checkout eeaea12e050a90b815d11d030e943e7b3181da21")
        .add_local_dir(ROOT / "worldsync", "/work/worldsync", ignore=["__pycache__"])
        .add_local_file(ROOT / "reproduction/lingbot-weights.json", "/work/reproduction/lingbot-weights.json")
        .add_local_file(ROOT / "reproduction/state.json", "/work/reproduction/state.json")
        .add_local_dir(ROOT / "external/PlayWorld/data", "/work/external/PlayWorld/data", ignore=["*.mp4"]))
else:
    image = None
    credentials = modal.Secret.from_dict({})
    weights = modal.Volume.from_name("worldsync-lingbot-weights")
    results = modal.Volume.from_name("worldsync-lingbot-results", create_if_missing=True)

app = modal.App("worldsync-lingbot-fourdim")

COMMON = ["--player-retries", "2", "--text-prompt", "both", "--offload-text-encoder"]
ARMS = {
    "stock": [],
    "budget": ["--budget-memory", "--budget-rope", "native", "--budget-retrieved", "10"],
}
MAX_CHUNKS = {"gc": 40, "if": 40, "outsight": 40, "insight": 60}  # insight: 59.8 s stationary WAIT, no player calls
CONCURRENCY = 8


@app.function(image=image, gpu="H200", cpu=8, memory=196608, timeout=3300, secrets=[credentials],
              volumes={"/cache": weights, "/results": results}, retries=0, max_containers=CONCURRENCY)
def generate(split: str, task: str, arm: str, seed: int):
    import shutil
    import subprocess
    import time
    weights.reload()
    target = Path("/work/external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast")
    target.parent.mkdir(exist_ok=True, parents=True)
    if not target.exists():
        target.symlink_to("/cache/14b-causal-fast")
    identifier = f"fourdim-{arm}-{split}-{task}-seed{seed}-{time.time_ns()}"
    output = Path("/work/outputs") / identifier
    outcome = None
    try:
        outcome = subprocess.run([sys.executable, "-u", "-m", "worldsync.lingbot_rollout", "--split", split,
            "--task-id", task, "--seed", str(seed), "--max-chunks", str(MAX_CHUNKS[split]), *COMMON, *ARMS[arm],
            "--output", str(output)], cwd="/work")
    finally:
        if output.exists():
            shutil.copytree(output, Path("/results") / identifier,
                            ignore=shutil.ignore_patterns("chunk-*.mp4", "observation-*.jpg"))
            results.commit()
    return {"identifier": identifier, "split": split, "task": task, "arm": arm, "seed": seed,
            "returncode": None if outcome is None else outcome.returncode}


@app.function(image=image, cpu=4, memory=8192, timeout=1200, secrets=[credentials], volumes={"/results": results})
def score(generated: dict):
    import subprocess
    if generated is None or generated.get("returncode") or "identifier" not in generated:
        return {**(generated or {}), "judge": None, "error": "generation failed or produced no artifacts"}
    results.reload()
    output = Path("/results") / generated["identifier"]
    if not (output / "rollout.mp4").is_file():
        return {**generated, "judge": None, "error": "no rollout.mp4"}
    data = Path("/work/external/PlayWorld/data")
    dataset = data / generated["split"] / "data.json"
    record = next(t for t in json.loads(dataset.read_text()) if t["task_id"] == generated["task"])
    outcome = subprocess.run([sys.executable, "-u", "-m", "worldsync.judge", "--dataset", str(dataset),
        "--task-id", generated["task"], "--video", str(output / "rollout.mp4"),
        "--reference-image", str(data / record["image_path"]), "--world-model", "LingBot-World2 14B causal-fast",
        "--output", str(output / "judge.json")], cwd="/work")
    results.commit()
    run = json.loads((output / "run.json").read_text())
    summary = {**generated, "generated_seconds": run.get("generated_seconds"), "termination": run.get("termination"),
               "elapsed_s": run.get("elapsed_s"), "peak_gb": (run.get("peak_allocated_bytes") or 0) / 1e9,
               "n_actions": len(run["actions"]), "player_retries": len(run.get("player_retry_events", []))}
    if outcome.returncode:
        return {**summary, "judge": None, "error": "judge failed"}
    return {**summary, "judge": json.loads((output / "judge.json").read_text())}


@app.local_entrypoint()
def main(seed: int = 42, arms: str = "stock,budget", per_dimension: int = 12, dimensions: str = "gc,if,insight,outsight",
         tag: str = ""):
    from worldsync.fourdim import select_cases
    chosen = [c for c in select_cases(ROOT / "external/PlayWorld/data", per_dimension) if c[0] in dimensions.split(",")]
    # Job order: dimension by dimension (least-tested dimensions first, so protocol problems surface early).
    priority = {"if": 0, "insight": 1, "outsight": 2, "gc": 3}
    chosen.sort(key=lambda case: priority[case[0]])
    jobs = [(split, task, arm, seed) for (split, task) in chosen for arm in arms.split(",")]
    out = ROOT / "outputs/lingbot-reproduction" / f"fourdim-seed{seed}{tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    report, handles = [], []
    for generated in generate.starmap(jobs, return_exceptions=True, order_outputs=False):
        generated = generated if isinstance(generated, dict) else {"error": repr(generated)}
        report.append(generated)
        (out / "generated.json").write_text(json.dumps(report, indent=2) + "\n")
        if "identifier" in generated:
            handles.append(score.spawn(generated))
        print("generated", generated.get("arm"), generated.get("split"), generated.get("task"),
              generated.get("returncode"), generated.get("error", ""), flush=True)
    final = []
    for handle in handles:
        try:
            final.append(handle.get())
        except Exception as error:  # keep going; failures are reported
            final.append({"error": repr(error)})
        (out / "fourdim-results.json").write_text(json.dumps(final, indent=2) + "\n")
    for r in final:
        j = r.get("judge") or {}
        print(r.get("arm"), r.get("split"), r.get("task"), j.get("reported_score"), (j.get("gate") or {}).get("verdict"),
              r.get("termination"), r.get("error", ""), flush=True)
