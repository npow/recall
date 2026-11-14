"""Run one complete control-calibration episode before freezing a local baseline."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worldsync.lingbot_rollout import PROTOCOL_ID
from worldsync.progress import publish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-pid", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "pilot.json"
    if report_path.exists():
        parser.error("pilot output already exists; preserve prior attempts")
    report = {"protocol_id": PROTOCOL_ID, "scope": "control calibration; excluded from baseline aggregate",
              "status": "queued", "published_score_reproduced": False, "hillclimbing_allowed": False,
              "episode": "gc/GC007", "output": str(output.relative_to(ROOT))}

    def save(status):
        report["status"] = status
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        publish(report, "lingbot-progress.json")

    save("waiting_for_prior_calibration_to_release_gpu")
    deadline = time.monotonic() + 7200
    while args.wait_for_pid:
        command = Path(f"/proc/{args.wait_for_pid}/cmdline")
        if not command.exists() or b"worldsync.lingbot_rollout" not in command.read_bytes():
            break
        if time.monotonic() > deadline:
            save("stopped_wait_timeout")
            raise SystemExit("Prior rollout still active; no second GPU process launched")
        time.sleep(5)
    episode = output / "episode"
    save("corrected_controller_complete_episode_running")
    with (output / "rollout.log").open("w") as log:
        result = subprocess.run([sys.executable, "-u", "-m", "worldsync.lingbot_rollout",
            "--split", "gc", "--task-id", "GC007", "--max-chunks", "40", "--output", str(episode)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        save("stopped_rollout_failure")
        raise SystemExit(result.returncode)
    run = json.loads((episode / "run.json").read_text())
    report["generated_seconds"] = run.get("generated_seconds")
    report["commanded_turn_actions"] = sum(a["action"] in {"LEFT", "RIGHT"} for a in run["actions"])
    if run["status"] != "completed":
        save("stopped_incomplete_rollout")
        raise SystemExit(1)
    save("scoring_complete_calibration_episode")
    data = ROOT / "external/PlayWorld/data"
    task = next(t for t in json.loads((data / "gc/data.json").read_text()) if t["task_id"] == "GC007")
    with (output / "judge.log").open("w") as log:
        result = subprocess.run([sys.executable, "-u", "-m", "worldsync.judge",
            "--dataset", str(data / "gc/data.json"), "--task-id", "GC007",
            "--video", str(episode / "rollout.mp4"), "--reference-image", str(data / task["image_path"]),
            "--world-model", "LingBot-World2 14B causal-fast stock", "--output", str(episode / "judge.json")],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        save("stopped_judge_failure")
        raise SystemExit(result.returncode)
    judged = json.loads((episode / "judge.json").read_text())
    report["local_calibration_score"] = judged.get("reported_score")
    report["gate"] = judged.get("gate")
    save("calibration_scored_pending_visual_control_review")


if __name__ == "__main__":
    main()
