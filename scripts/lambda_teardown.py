"""Finish a run without idle billing (run LOCALLY, needs LAMBDA_API_KEY in the environment).

    python scripts/lambda_teardown.py --remote final --label final1 --lists outputs/lambda/final1 \
        --dest <worktree>/outputs/final1 --log outputs/lambda/teardown.jsonl

Every cycle: (1) run the tail rebalance (idle instances take queued jobs from the busiest ones);
(2) for every instance that is idle and got nothing, pull its results, verify that every job it completed has
run.json and rollout.mp4 locally, then terminate it. The last instance is never terminated here: the loop exits
when one instance is left and idle, so it can serve reruns. Only instances named worldsync-optb-* are touched.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lambda_cloud
import lambda_fleet as fleet
import lambda_tail

HERE = Path(__file__).resolve().parent


def fleet_instances():
    rows = lambda_cloud.call("instances")["data"]
    return {row["ip"]: row for row in rows if (row.get("name") or "").startswith("worldsync-optb-") and row["status"] == "active"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", required=True); parser.add_argument("--label", required=True)
    parser.add_argument("--lists", type=Path, required=True); parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True); parser.add_argument("--interval", type=int, default=180)
    parser.add_argument("--keep", type=int, default=1, help="how many instances to leave running for later work")
    args = parser.parse_args()

    def log(**row):
        row["time"] = time.strftime("%H:%M:%S")
        with open(args.log, "a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    while True:
        instances = fleet_instances()
        ips = sorted(instances)
        if not ips:
            log(event="no_instances")
            return
        plan = subprocess.run([sys.executable, str(HERE / "lambda_tail.py"), "--remote", args.remote, "--label", args.label,
                               "--lists", str(args.lists), "--apply", *ips], capture_output=True, text=True)
        try:
            tail = json.loads(plan.stdout[plan.stdout.index("{"):plan.stdout.rindex("}") + 1])
        except ValueError:
            log(event="tail_failed", detail=plan.stderr[-300:])
            time.sleep(args.interval)
            continue
        received = {ip for ip, moved in tail["moves"].items() if moved}
        candidates = [ip for ip in tail["idle"] if ip not in received]
        log(event="cycle", active=len(ips), idle=tail["idle"], received=sorted(received), remaining_minutes=tail["remaining_minutes"])
        for ip in candidates:
            if len(fleet_instances()) <= args.keep:
                log(event="last_instance_idle", ip=ip, id=instances[ip]["id"])
                return
            done, failed, inflight, idle = lambda_tail.state(ip, args.remote, args.label)
            if not idle or inflight:
                continue
            pull = argparse.Namespace(label=args.label, dest=args.dest, ips=[ip], remote=args.remote)
            fleet.pull_dev(pull)
            missing = [name for name in done if not ((args.dest / name / "run.json").exists() and (args.dest / name / "rollout.mp4").exists())]
            if missing:
                log(event="not_synced", ip=ip, missing=missing[:10])
                continue
            lambda_cloud.call("instance-operations/terminate", {"instance_ids": [instances[ip]["id"]]})
            log(event="terminated", ip=ip, id=instances[ip]["id"], name=instances[ip]["name"], jobs_completed_there=len(done),
                persistent_failures_there=sorted(failed))
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
