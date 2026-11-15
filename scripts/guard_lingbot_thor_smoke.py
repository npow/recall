"""Guard an isolated two-chunk Thor child with a16GiB physical-memory reserve."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RESERVE = 16 * 1024 ** 3
APP_ESTIMATE = 72_758_273_536 + 5_000_000_000 + 6_000_000_000


def available():
    line = next(line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    return int(line.split()[1]) * 1024


def main():
    output = ROOT / "outputs/lingbot-reproduction" / ("thor-two-chunk-smoke-" + str(time.time_ns()))
    output.mkdir()
    before = available()
    report = {"scope": "supervised one-shot no-archive local memory smoke", "available_before_bytes": before,
              "app_estimate_bytes": APP_ESTIMATE, "system_reserve_bytes": RESERVE,
              "output": str(output), "status": "not_launched"}
    gpu_processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                   capture_output=True, text=True, check=True).stdout.strip()
    if gpu_processes:
        report["status"] = "gpu_busy_not_launched"
    elif before < APP_ESTIMATE + RESERVE:
        report["status"] = "budget_rejected"
    else:
        with (output / "child.log").open("w") as log, (output / "supervisor-samples.jsonl").open("w") as samples:
            process = subprocess.Popen([str(ROOT / ".venv-lingbot/bin/python"), "-u", "-m", "worldsync.lingbot_thor_smoke",
                                        "--output", str(output)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            report["status"] = "running"
            report["pid"] = process.pid
            (output / "supervisor.json").write_text(json.dumps(report, indent=2) + "\n")
            while process.poll() is None:
                memory = available()
                samples.write(json.dumps({"monotonic_s": time.monotonic(), "mem_available_bytes": memory}) + "\n")
                samples.flush()
                if memory < RESERVE:
                    report["status"] = "reserve_abort"
                    report["reserve_abort_available_bytes"] = memory
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break
                time.sleep(.25)
            report["returncode"] = process.wait()
            if report["status"] != "reserve_abort":
                report["status"] = "completed" if report["returncode"] == 0 else "failed"
    (output / "supervisor.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
