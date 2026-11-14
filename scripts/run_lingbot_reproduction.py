"""Await verified LingBot weights and run the official native smoke case."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worldsync.progress import publish


def main():
    output = ROOT / "outputs/lingbot-reproduction"
    output.mkdir(parents=True, exist_ok=True)
    progress = json.loads((ROOT / "reproduction/lingbot-progress.json").read_text())
    def status(value):
        progress["status"] = value
        publish(progress, "lingbot-progress.json")
    manifest = ROOT / "reproduction/lingbot-weights.json"
    deadline = time.monotonic() + 6 * 3600
    while not manifest.exists():
        if time.monotonic() > deadline:
            status("checkpoint_wait_timed_out")
            raise SystemExit("Checkpoint verification did not complete")
        time.sleep(15)
    status("native_smoke_running")
    directory = output / "native-smoke"
    if directory.exists() and any(directory.iterdir()):
        directory = output / f"native-smoke-attempt-{time.time_ns()}"
    with (output / "native-smoke.log").open("a") as log:
        result = subprocess.run([sys.executable, "-u", "-m", "worldsync.lingbot_native",
            "--output", str(directory)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    report_path = directory / "run.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    if result.returncode or report.get("status") != "completed":
        status("native_smoke_failed")
        raise SystemExit(result.returncode or 1)
    progress["native_smoke"] = {key: report[key] for key in ("frames", "frame_shape", "elapsed_s", "peak_allocated_bytes", "video_sha256")}
    status("native_smoke_passed_verifying_adapter_parity")
    parity = output / f"adapter-parity-{time.time_ns()}"
    with (output / "adapter-parity.log").open("a") as log:
        result = subprocess.run([sys.executable, "-u", "-m", "worldsync.lingbot_rollout",
            "--split", "gc", "--task-id", "GC007", "--max-chunks", "1", "--fixed-actions", "WAIT",
            "--output", str(parity)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        status("adapter_parity_run_failed")
        raise SystemExit(result.returncode)
    import numpy as np
    native_pixels = np.load(directory / "pixels.npy")
    hooked_pixels = np.load(parity / "pixels-000.npy")
    identical = native_pixels.shape == hooked_pixels.shape and np.array_equal(native_pixels, hooked_pixels)
    parity_report = {"status": "passed" if identical else "failed",
        "scope": "fixed WAIT native generation vs callback adapter; no benchmark score",
        "native_shape": list(native_pixels.shape), "adapter_shape": list(hooked_pixels.shape),
        "pixel_identical": identical, "published_score_reproduced": False}
    (ROOT / "reproduction/lingbot-adapter-verification.json").write_text(json.dumps(parity_report, indent=2) + "\n")
    if not identical:
        status("adapter_pixels_differ_from_native")
        raise SystemExit("Source/adapter pixel parity failed; full suite stays stopped")
    progress["adapter_pixel_parity"] = True
    status("native_and_adapter_parity_passed_starting_full_suite")
    with (output / "suite.log").open("a") as log:
        result = subprocess.run([sys.executable, "-u", "-m", "worldsync.evaluate", "--model", "lingbot",
            "--output", str(output / "suite"), "--publish-status", "--judge"], cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
