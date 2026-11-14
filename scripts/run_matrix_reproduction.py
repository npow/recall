"""Await verified weights, test native generation, then run the full stock suite.

Stops on failure; never changes settings or launches optimization.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worldsync.matrix_suite import publish


def main():
    output = ROOT / "outputs/matrix-reproduction"
    output.mkdir(parents=True, exist_ok=True)
    def status(phase):
        value = {"workflow": "stock Matrix reproduction", "status": phase,
            "published_score_reproduced": False, "hillclimbing_allowed": False,
            "scope": "native smoke test followed by full 171-case stock generation reconstruction"}
        (output / "workflow.json").write_text(json.dumps(value, indent=2) + "\n")
        publish(value)
    status("waiting_for_verified_checkpoints")
    deadline = time.monotonic() + 4 * 3600
    manifest = ROOT / "reproduction/matrix-weights.json"
    while not manifest.is_file():
        if time.monotonic() > deadline:
            status("checkpoint_wait_timed_out")
            raise SystemExit("Verified checkpoint inventory did not arrive")
        time.sleep(15)
    weights = json.loads(manifest.read_text())
    if weights["revision"] != "d4c84ef2b8e9028a00a5d58b6be2df04f1cb6042":
        raise ValueError("unexpected checkpoint revision")
    checkpoint = ROOT / "external/Matrix-Game/checkpoints/Matrix-Game-3.0"
    for file in weights["files"]:
        if (checkpoint / file["path"]).stat().st_size != file["bytes"]:
            raise ValueError("checkpoint inventory is incomplete")
    smoke = output / "native-smoke"
    prior = smoke / "run.json"
    passed = prior.exists() and json.loads(prior.read_text()).get("status") == "completed"
    if smoke.exists() and any(smoke.iterdir()) and not passed:
        smoke = output / f"native-smoke-retry-{time.time_ns()}"
    if not passed:
        status("native_smoke_running")
        with (output / "native-smoke.log").open("a") as log:
            result = subprocess.run([sys.executable, "-u", "-m", "worldsync.matrix_baseline",
                "--split", "gc", "--task-id", "GC007", "--max-chunks", "1", "--fixed-actions", "WAIT",
                "--output", str(smoke)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status("native_smoke_failed")
            raise SystemExit(result.returncode)
    run = json.loads((smoke / "run.json").read_text())
    if run.get("status") != "completed" or run.get("generated_frames") != 57:
        status("native_smoke_incomplete")
        raise SystemExit("Native smoke did not produce the expected 57 frames")
    status("native_smoke_passed_starting_full_stock_suite")
    with (output / "suite.log").open("a") as log:
        result = subprocess.run([sys.executable, "-u", "-m", "worldsync.matrix_suite",
            "--output", str(output / "suite"), "--publish-status"], cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
