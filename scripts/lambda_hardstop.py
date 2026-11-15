"""Safety net: terminate every worldsync-optb-* instance at a wall-clock time (run LOCALLY, LAMBDA_API_KEY in env).

    nohup python scripts/lambda_hardstop.py --at 12:00 --log outputs/lambda/hardstop.jsonl &

Only instances whose name starts with worldsync-optb- are touched. Kill this process to cancel.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lambda_cloud


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--at", required=True, help="local time HH:MM, today")
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    while time.strftime("%H:%M") < args.at:
        time.sleep(20)
    rows = [row for row in lambda_cloud.call("instances")["data"] if (row.get("name") or "").startswith("worldsync-optb-")]
    ids = [row["id"] for row in rows if row["status"] not in ("terminating", "terminated")]
    if ids:
        lambda_cloud.call("instance-operations/terminate", {"instance_ids": ids})
    with open(args.log, "a") as stream:
        stream.write(json.dumps({"time": time.strftime("%H:%M:%S"), "event": "hard_stop",
                                 "terminated": [(row["name"], row["id"]) for row in rows if row["id"] in ids]}) + "\n")


if __name__ == "__main__":
    main()
