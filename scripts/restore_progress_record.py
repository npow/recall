"""Restore a 'completed' progress record that was overwritten by a later 'skipped_already_complete' record.

    MODAL_PROFILE=nissan-pow python scripts/restore_progress_record.py --label full1 --arm stock --split if --task IF030

A job popped a second time (after two queue rebuilds) is skipped by the worker because its episode directory already
holds a completed attempt, and the skip record has the same filename as the completion record. The episode itself is
intact; this rewrites the record as completed so status and the scoring pipeline see it. Verifies the episode's
run.json says completed first.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

VOLUME = "worldsync-lingbot-results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="full1")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        directory = f"adaptive/{args.label}/{args.arm}/{args.split}-{args.task}-seed42__a0"
        record = f"adaptive/{args.label}/progress/{args.arm}-{args.split}-{args.task}-seed42.json"
        for remote in (f"{directory}/run.json", record):
            subprocess.run(["modal", "volume", "get", VOLUME, remote, str(tmp) + "/", "--force"], check=True,
                           capture_output=True)
        run = json.loads((tmp / "run.json").read_text())
        if run.get("status") != "completed":
            raise SystemExit(f"episode not completed: {run.get('status')}")
        progress = json.loads((tmp / Path(record).name).read_text())
        if progress.get("status") == "completed":
            print("record already completed; nothing to do")
            return
        progress.update({"status": "completed", "result_directory": directory,
                         "generated_seconds": run.get("generated_seconds"), "termination": run.get("termination"),
                         "note": "restored by the harness: the original completed record was overwritten by a later "
                                 "skipped_already_complete record; episode directory and run.json are intact"})
        (tmp / Path(record).name).write_text(json.dumps(progress, indent=1) + "\n")
        subprocess.run(["modal", "volume", "put", VOLUME, str(tmp / Path(record).name), record, "--force"],
                       check=True, capture_output=True)
        print("restored", record, "->", directory)


if __name__ == "__main__":
    main()
