"""Memory-LoRA harness validation on one Modal H200 (profile nissan-pow). No agent/API calls.

  MODAL_PROFILE=nissan-pow modal run scripts/modal_lingbot_train.py --steps 40
  MODAL_PROFILE=nissan-pow modal run scripts/modal_lingbot_train.py --manifest /data/manifest.jsonl --steps 500

Trains on synthetic pan clips unless --manifest is given (manifest lives on the
`worldsync-train-data` volume). Adapter and train.json land in the results volume.
"""
import json
from pathlib import Path
import sys
import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if modal.is_local():
    from scripts.modal_lingbot_benchmark import image as base_image, weights, results
    image = base_image.add_local_file(ROOT / "scripts/train_lingbot_memory.py", "/work/scripts/train_lingbot_memory.py")
else:
    image = None
    weights = modal.Volume.from_name("worldsync-lingbot-weights")
    results = modal.Volume.from_name("worldsync-lingbot-results")
train_data = modal.Volume.from_name("worldsync-train-data", create_if_missing=True)

app = modal.App("worldsync-lingbot-train")


@app.function(image=image, gpu="H200", cpu=8, memory=196608, timeout=7200, max_containers=1,
              volumes={"/cache": weights, "/results": results, "/data": train_data})
def train(steps: int = 40, tasks: str = "GC001,GC002,GC003,GC004", heldout: str = "GC005", manifest: str = "",
          lr: float = 1e-4, rank: int = 32, loss: str = "velocity", parity: bool = True, rollout_smoke: bool = True,
          content_ok_only: bool = False, heldout_count: int = 0, max_train: int = 0):
    import shutil
    import subprocess
    import time
    weights.reload()
    train_data.reload()
    target = Path("/work/external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast")
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        target.symlink_to("/cache/14b-causal-fast")
    identifier = f"train-lora-{time.time_ns()}"
    output = Path("/work/outputs") / identifier
    command = [sys.executable, "-u", "/work/scripts/train_lingbot_memory.py", "--output", str(output), "--steps", str(steps),
        "--lr", str(lr), "--rank", str(rank), "--loss", loss]
    if heldout:
        command += ["--heldout-tasks", *heldout.split(",")]
    if content_ok_only:
        command.append("--content-ok-only")
    command += ["--heldout-count", str(heldout_count), "--max-train", str(max_train)]
    if tasks:
        command += ["--synthetic-tasks", *tasks.split(",")]
    if manifest:
        command += ["--manifest", manifest]
    if parity:
        command.append("--parity")
    summary = {"result_directory": identifier}
    try:
        outcome = subprocess.run(command, cwd="/work", env={**__import__("os").environ, "PYTHONPATH": "/work"})
        summary["train_returncode"] = outcome.returncode
        if outcome.returncode == 0 and rollout_smoke:
            smoke = Path("/work/outputs") / f"{identifier}-rollout"
            run = subprocess.run([sys.executable, "-u", "-m", "worldsync.lingbot_rollout", "--split", "gc", "--task-id", "GC007",
                "--max-chunks", "2", "--native-fixed-controls", "--fixed-actions", "WAIT", "RIGHT", "--controlled-ablation",
                "--lora", str(output / "lora.pt"), "--output", str(smoke)], cwd="/work")
            summary["rollout_smoke_returncode"] = run.returncode
            if smoke.exists():
                shutil.copytree(smoke, Path("/results") / f"{identifier}-rollout")
    finally:
        if output.exists():
            shutil.copytree(output, Path("/results") / identifier)
        results.commit()
    if (output / "train.json").exists():
        summary["train"] = json.loads((output / "train.json").read_text())
    return summary


@app.local_entrypoint()
def main(steps: int = 40, tasks: str = "GC001,GC002,GC003,GC004", heldout: str = "GC005", manifest: str = "",
         lr: float = 1e-4, rank: int = 32, loss: str = "velocity", parity: bool = True, rollout_smoke: bool = True,
         content_ok_only: bool = False, heldout_count: int = 0, max_train: int = 0):
    summary = train.remote(steps, tasks, heldout, manifest, lr, rank, loss, parity, rollout_smoke, content_ok_only, heldout_count, max_train)
    destination = ROOT / "outputs" / "lingbot-train"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{summary['result_directory']}.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    train_report = summary.get("train", {})
    print(json.dumps({"saved": str(path), "returncodes": {k: v for k, v in summary.items() if k.endswith("returncode")},
        "parity_vs_stock": train_report.get("parity_vs_stock"), "train": train_report.get("train"),
        "eval_before": train_report.get("eval_before"), "eval_after": train_report.get("eval_after"),
        "lora_roundtrip": train_report.get("lora_roundtrip")}, indent=2))
