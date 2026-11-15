"""Rebalance the remaining jobs of a dev batch across instances by chunk count (run LOCALLY).

    python scripts/lambda_rebalance.py --jobs JOBS.jsonl --remote director --label director-lambda \
        --max-stage 8 --out-dir outputs/lambda/rebalance IP [IP ...]

Reads which job ids are completed on every instance, drops them and every job above --max-stage, then deals the
rest out stage by stage (lowest stage first), always giving the next-largest job to the instance with the
fewest chunks so far. The job an instance is running right now stays first on that instance, so a restart
loses only that job's elapsed time. Writes one job list per instance; start each with
`lambda_fleet.py --remote R start --jobs FILE --plain LABEL --no-player --shards 1 IP`.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lambda_fleet as fleet


def remote_state(ip, remote, label):
    script = (f"cd ~/{remote}; for d in results/dev/{label}/*/; do "
              "python3 -c \"import json,sys; print(json.load(open(sys.argv[1]+'run.json')).get('status'), sys.argv[1])\" $d 2>/dev/null; done; "
              f"echo RUNNING $(ls -t outputs/dev/{label} 2>/dev/null | head -n 1)\n")
    out = fleet.ssh(ip, script, capture=True, check=False).stdout
    done, running = set(), None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "completed":
            done.add(parts[1].rstrip("/").rsplit("/", 1)[-1])
        elif parts and parts[0] == "RUNNING" and len(parts) == 2:
            running = parts[1]
    return done, running


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-stage", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--keep-in-flight-stage", type=int, default=3,
                        help="an in-flight job stays first on its instance only if its stage is at most this")
    parser.add_argument("ips", nargs="+")
    args = parser.parse_args()
    jobs = [json.loads(line) for line in args.jobs.read_text().splitlines() if line.strip()]
    done, running = set(), {}
    for ip in args.ips:
        finished, current = remote_state(ip, args.remote, args.label)
        done |= finished
        running[ip] = current if current not in finished else None
    remaining = [job for job in jobs if job["id"] not in done and job.get("stage", 0) <= args.max_stage]
    by_id = {job["id"]: job for job in remaining}
    plan = {ip: [] for ip in args.ips}
    load = {ip: 0 for ip in args.ips}
    for ip, current in running.items():  # keep the in-flight job where it is
        if current in by_id and by_id[current].get("stage", 0) <= args.keep_in_flight_stage:
            plan[ip].append(by_id.pop(current))
            load[ip] += plan[ip][-1].get("chunks", 1)
    for stage in sorted({job.get("stage", 0) for job in by_id.values()}):
        for job in sorted((job for job in by_id.values() if job.get("stage", 0) == stage), key=lambda job: -job.get("chunks", 1)):
            ip = min(args.ips, key=lambda candidate: load[candidate])
            plan[ip].append(job)
            load[ip] += job.get("chunks", 1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"completed_everywhere": len(done), "remaining": len(remaining), "dropped_above_stage": args.max_stage, "instances": {}}
    for ip, rows in plan.items():
        path = args.out_dir / f"jobs-{ip}.jsonl"
        path.write_text("".join(json.dumps(job) + "\n" for job in rows))
        stages = {}
        for job in rows:
            stages[job.get("stage", 0)] = stages.get(job.get("stage", 0), 0) + job.get("chunks", 1)
        summary["instances"][ip] = {"jobs": len(rows), "chunks": load[ip], "chunks_by_stage": stages, "in_flight": running[ip], "file": str(path)}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
