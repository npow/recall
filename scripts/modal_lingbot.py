"""Run the LingBot gate and one verification case on Modal. Never runs the full suite.

    modal run --detach scripts/modal_lingbot.py [--case gc:GC007]

Steps: weight-size check, native smoke, native-vs-adapter pixel parity, then one
adaptive case (Haiku player, Claude judge diagnostics) through the suite code.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "/root/proj"
app = modal.App("worldsync-lingbot-gate")
weights = modal.Volume.from_name("worldsync-lingbot-weights")
results = modal.Volume.from_name("worldsync-lingbot-results", create_if_missing=True)


def _key():
    # Read locally only; passed as an ephemeral secret, never stored in the workspace.
    env = os.environ.get("ANTHROPIC_API_KEY")
    path = ROOT / ".env.local"
    if not env and path.is_file():
        for line in path.read_text().splitlines():
            name, _, value = line.partition("=")
            if name == "ANTHROPIC_API_KEY":
                env = value.strip()
    return env or ""


image = (modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.6.0", "torchvision==0.21.0")
    .pip_install("transformers==4.51.3", "numpy==1.26.4", "tokenizers>=0.20.3", "opencv-python-headless",
        "diffusers", "accelerate", "tqdm", "imageio[ffmpeg]", "easydict", "ftfy", "scipy", "omegaconf",
        "einops", "safetensors", "sentencepiece", "Pillow", "anthropic", "pydantic", "huggingface_hub")
    .add_local_dir(ROOT / "worldsync", f"{PROJECT}/worldsync", ignore=lambda path: "__pycache__" in Path(path).parts)
    .add_local_file(ROOT / "reproduction/lingbot-weights.json", f"{PROJECT}/reproduction/lingbot-weights.json")
    .add_local_dir(ROOT / "external/LingBot-World2", f"{PROJECT}/external/LingBot-World2",
        ignore=lambda path: "checkpoints" in Path(path).parts or "__pycache__" in Path(path).parts)
    .add_local_dir(ROOT / "external/PlayWorld", f"{PROJECT}/external/PlayWorld", ignore=lambda path: "__pycache__" in Path(path).parts))


def sh(command, log, **kwargs):
    with open(log, "a") as handle:
        return subprocess.run(command, cwd=PROJECT, stdout=handle, stderr=subprocess.STDOUT, **kwargs).returncode


@app.function(image=image, gpu="H200", volumes={"/weights": weights, "/results": results},
    secrets=[modal.Secret.from_dict({"ANTHROPIC_API_KEY": _key()})], timeout=4 * 3600)
def gate_and_one_case(run_id: str, case: str = "gc:GC007"):
    import numpy as np
    out = Path("/results") / run_id
    out.mkdir(parents=True, exist_ok=True)
    report = {"run_id": run_id, "case": case, "scope": "pipeline verification; not a benchmark score", "steps": {}}

    def save(step=None, value=None):
        if step:
            report["steps"][step] = value
        (out / "gate.json").write_text(json.dumps(report, indent=2) + "\n")
        results.commit()

    def fail(step, detail):
        save(step, {"status": "failed", "detail": detail})
        raise SystemExit(f"{step} failed: {detail}")

    import torch
    report["gpu"] = torch.cuda.get_device_name(0)
    report["torch"] = torch.__version__
    # 1. Weights: sizes must match the verified manifest.
    raw = Path(f"{PROJECT}/external/LingBot-World2/checkpoints")
    raw.mkdir(parents=True, exist_ok=True)
    link = raw / "lingbot-world-v2-14b-causal-fast"
    if not link.exists():
        link.symlink_to("/weights/14b-causal-fast")
    manifest = json.loads(Path(f"{PROJECT}/reproduction/lingbot-weights.json").read_text())
    bad = [item["path"] for item in manifest["files"] if not (link / item["path"]).exists() or (link / item["path"]).stat().st_size != item["bytes"]]
    if bad:
        fail("weights", f"missing or wrong size: {bad[:5]}")
    save("weights", {"status": "passed", "files": len(manifest["files"])})
    # 2. Native smoke.
    native = out / "native-smoke"
    code = sh([sys.executable, "-u", "-m", "worldsync.lingbot_native", "--output", str(native)], out / "native-smoke.log")
    native_report = json.loads((native / "run.json").read_text()) if (native / "run.json").exists() else {}
    if code or native_report.get("status") != "completed":
        fail("native_smoke", f"exit {code}; see native-smoke.log")
    save("native_smoke", {"status": "passed", **{k: native_report[k] for k in ("frames", "frame_shape", "elapsed_s", "peak_allocated_bytes")}})
    # 3. Adapter parity against the native pixels.
    parity = out / "adapter-parity"
    code = sh([sys.executable, "-u", "-m", "worldsync.lingbot_rollout", "--split", "gc", "--task-id", "GC007",
        "--max-chunks", "1", "--fixed-actions", "WAIT", "--output", str(parity)], out / "adapter-parity.log")
    if code:
        fail("adapter_parity", f"adapter run exit {code}")
    a, b = np.load(native / "pixels.npy"), np.load(parity / "pixels-000.npy")
    identical = a.shape == b.shape and np.array_equal(a, b)
    save("adapter_parity", {"status": "passed" if identical else "failed", "native_shape": list(a.shape), "adapter_shape": list(b.shape),
        "pixel_identical": bool(identical), "max_abs_diff": int(np.abs(a.astype(int) - b.astype(int)).max()) if a.shape == b.shape else None})
    if not identical:
        raise SystemExit("adapter pixels differ from native; stopping before the case")
    # 4. One adaptive case through the suite code (never the full suite).
    start = time.monotonic()
    code = sh([sys.executable, "-u", "-m", "worldsync.evaluate", "--model", "lingbot", "--only", case, "--judge",
        "--output", str(out / "suite")], out / "suite.log")
    suite = json.loads((out / "suite/suite.json").read_text()) if (out / "suite/suite.json").exists() else {}
    save("one_case", {"status": "passed" if code == 0 else "failed", "exit": code, "wall_s": round(time.monotonic() - start, 1),
        "runs": suite.get("runs"), "diagnostic_scores": suite.get("diagnostic_scores")})
    return report


@app.function(image=image, gpu="H200", volumes={"/weights": weights, "/results": results},
    secrets=[modal.Secret.from_dict({"ANTHROPIC_API_KEY": _key()})], timeout=4 * 3600, max_containers=3)  # account GPU limit is believed to be 3
def seed_case(run_id: str, seed: int, case: str = "gc:GC007"):
    """One scored episode at one seed, through the suite code; gate is assumed passed."""
    out = Path("/results") / run_id / f"{case.replace(':', '-')}-seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    raw = Path(f"{PROJECT}/external/LingBot-World2/checkpoints")
    raw.mkdir(parents=True, exist_ok=True)
    link = raw / "lingbot-world-v2-14b-causal-fast"
    if not link.exists():
        link.symlink_to("/weights/14b-causal-fast")
    start = time.monotonic()
    code = sh([sys.executable, "-u", "-m", "worldsync.evaluate", "--model", "lingbot", "--only", case, "--judge",
        "--seed", str(seed), "--output", str(out / "suite")], out / "suite.log")
    suite = json.loads((out / "suite/suite.json").read_text()) if (out / "suite/suite.json").exists() else {}
    summary = {"seed": seed, "exit": code, "wall_s": round(time.monotonic() - start, 1), "runs": suite.get("runs")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    results.commit()
    return summary


@app.local_entrypoint()
def seeds(seeds: str = "1,2,3,4,5", case: str = "gc:GC007"):
    """Fan out one episode per seed on separate GPUs. Never the full suite."""
    if not _key():
        raise SystemExit("ANTHROPIC_API_KEY missing locally")
    run_id = f"seeds-{int(time.time())}"
    print("run_id", run_id, flush=True)
    for summary in seed_case.starmap([(run_id, int(s), case) for s in seeds.split(",")]):
        print(json.dumps(summary), flush=True)


@app.local_entrypoint()
def cohort(seed: int = 42, label: str = "stock"):
    """The predeclared 16-episode tuning cohort at one seed, 3 GPUs at a time."""
    if not _key():
        raise SystemExit("ANTHROPIC_API_KEY missing locally")
    split = json.loads((ROOT / "reproduction/episode-split.json").read_text())
    cases = [f"{c['split']}:{c['task_id']}" for c in split["predeclared_tuning_cohort_complete_episodes"]]
    cases.sort(key=lambda c: c.startswith("insight"), reverse=True)  # longest (60-chunk) cases first
    run_id = f"cohort-{label}-s{seed}-{int(time.time())}"
    print("run_id", run_id, len(cases), "cases", flush=True)
    for summary in seed_case.starmap([(run_id, seed, c) for c in cases]):
        print(json.dumps({k: summary[k] for k in ("seed", "exit", "wall_s")}), flush=True)


@app.local_entrypoint()
def main(case: str = "gc:GC007"):
    if not _key():
        raise SystemExit("ANTHROPIC_API_KEY missing locally")
    run_id = f"gate-{int(time.time())}"
    print("run_id", run_id)
    print(json.dumps(gate_and_one_case.remote(run_id, case), indent=2))
