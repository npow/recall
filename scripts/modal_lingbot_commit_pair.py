"""Launch ONLY an explicitly frozen H200 stock/Commit-Forcing pair."""

import json
from pathlib import Path
import sys
import time
import traceback

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if modal.is_local():
    from scripts.modal_lingbot_benchmark import image as base_image, weights, results

    reference = "reproduction/task-memory-benchmark/GC010-seed42-turn90_wait_return/stock-run.json"
    image = base_image.add_local_file(ROOT / reference, "/work/" + reference)
else:
    image = None
    weights = modal.Volume.from_name("worldsync-lingbot-weights")
    results = modal.Volume.from_name("worldsync-lingbot-results")
app = modal.App("worldsync-lingbot-commit-pair")


@app.function(
    image=image,
    gpu="H200",
    cpu=8,
    memory=131072,
    timeout=2400,
    max_containers=1,
    volumes={"/cache": weights, "/results": results},
)
def pair(plan: dict):
    import os
    import shutil
    import subprocess

    weights.reload()
    raw = Path("/cache/14b-causal-fast")
    inventory = json.loads(Path("/work/reproduction/lingbot-weights.json").read_text())
    if any(
        (raw / row["path"]).stat().st_size != row["bytes"] for row in inventory["files"]
    ):
        raise ValueError("Verified checkpoint staging inventory changed")
    target = Path(
        "/work/external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast"
    )
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        target.symlink_to(raw)
    identifier = "commit-pair-GC010-seed42-" + str(time.time_ns())
    root = Path("/work/outputs") / identifier
    root.mkdir(parents=True)
    config = root / "plan.json"
    config.write_text(json.dumps(plan, indent=2) + "\n")
    try:
        with (root / "stdout.log").open("w") as log:
            outcome = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "worldsync.lingbot_commit_pair",
                    "--plan",
                    str(config),
                    "--output",
                    str(root / "pair"),
                ],
                cwd="/work",
                stdout=log,
                stderr=subprocess.STDOUT,
                env=dict(
                    os.environ,
                    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                    WORLDSYNC_LINGBOT_ATTENTION="sdpa",
                ),
            )
    finally:
        shutil.copytree(root, Path("/results") / identifier, dirs_exist_ok=True)
        results.commit()
    path = root / "pair/summary.json"
    summary = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "status": "failed",
            "error": "child stopped before summary; inspect stdout.log",
        }
    )
    runs = {}
    for arm in plan["arms"]:
        file = root / "pair" / arm / "run.json"
        if file.exists():
            runs[arm] = json.loads(file.read_text())
    return {
        "result_directory": identifier,
        "returncode": outcome.returncode,
        "summary": summary,
        "runs": runs,
    }


@app.local_entrypoint()
def main(plan: str, output: str):
    from worldsync.lingbot_commit_pair import validate_plan

    directory = ROOT / output
    directory.mkdir(parents=True, exist_ok=False)
    attempt = {"status": "preflight", "plan": plan, "launched": False}
    path = directory / "attempt.json"
    try:
        specification = json.loads((ROOT / plan).read_text())
        validate_plan(specification)
        attempt.update(
            status="submitted", launched=True, plan_sha256=specification["plan_sha256"]
        )
        path.write_text(json.dumps(attempt, indent=2) + "\n")
        result = pair.remote(specification)
        download_errors = []
        for arm, run in result.pop("runs").items():
            folder = directory / arm
            folder.mkdir()
            (folder / "run.json").write_text(json.dumps(run, indent=2) + "\n")
            if run.get("chunks"):
                try:
                    with (folder / "rollout.mp4").open("wb") as movie:
                        for part in results.read_file(
                            f"{result['result_directory']}/pair/{arm}/rollout.mp4"
                        ):
                            movie.write(part)
                except Exception as error:
                    download_errors.append({"arm": arm, "error": str(error)})
        result["artifact_download_errors"] = download_errors
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        attempt.update(
            status=result["summary"]["status"],
            result_directory=result["result_directory"],
        )
        if result["returncode"] or result["summary"]["status"] != "completed":
            raise RuntimeError("Pair failed; all artifacts retained")
    except BaseException:
        attempt.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        path.write_text(json.dumps(attempt, indent=2) + "\n")
