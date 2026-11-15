"""Split and monitor a final-run job list across Lambda instances (run LOCALLY).

    python scripts/lambda_final.py inspect --jobs JOBS.jsonl
    python scripts/lambda_final.py split --jobs JOBS.jsonl --out-dir outputs/lambda/final1 IP [IP ...]
    python scripts/lambda_final.py status --jobs JOBS.jsonl --dest <worktree>/outputs/final1 [--failed-dir DIR]

`split` deals jobs out in file order (the file's priority order is preserved inside every instance), always to
the instance with the smallest estimated chunk load. Chunk estimates: the job's `chunks` field if present,
otherwise 60 for insight (stationary, fixed length) and the measured full-run medians for adaptive episodes.
"""
import argparse
import collections
import json
from pathlib import Path

ESTIMATE = {"insight": 60, "gc": 9, "if": 5, "outsight": 6}
SECONDS_PER_CHUNK = {"insight": 10.0, "gc": 17.0, "if": 17.0, "outsight": 17.0}  # measured on H100 PCIe


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def chunks(job):
    return job.get("chunks") or ESTIMATE.get(job.get("split"), 8)


def seconds(job):
    return chunks(job) * SECONDS_PER_CHUNK.get(job.get("split"), 15.0) + 12  # + per-job overhead


def inspect(args):
    jobs = load(args.jobs)
    print(json.dumps({"jobs": len(jobs), "fields": sorted(jobs[0]),
                      "by_arm_split": {f"{arm}/{split}": count for (arm, split), count in
                                       sorted(collections.Counter((job["arm"], job["split"]) for job in jobs).items())},
                      "needs_player": dict(collections.Counter(str(job.get("needs_player")) for job in jobs)),
                      "estimated_gpu_hours": round(sum(seconds(job) for job in jobs) / 3600, 1),
                      "first_argv": jobs[0]["argv"]}, indent=1))


def split(args):
    jobs = load(args.jobs)
    plan = {ip: [] for ip in args.ips}
    load_seconds = {ip: 0.0 for ip in args.ips}
    for job in jobs:
        ip = min(args.ips, key=lambda candidate: load_seconds[candidate])
        plan[ip].append(job)
        load_seconds[ip] += seconds(job)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ip, rows in plan.items():
        (args.out_dir / f"jobs-{ip}.jsonl").write_text("".join(json.dumps(job) + "\n" for job in rows))
    print(json.dumps({ip: {"jobs": len(rows), "estimated_hours": round(load_seconds[ip] / 3600, 2)} for ip, rows in plan.items()}, indent=1))


def status(args):
    jobs = load(args.jobs)
    done = collections.Counter()
    total = collections.Counter()
    missing = []
    for job in jobs:
        key = f"{job['arm']}/{job['split']}"
        total[key] += 1
        try:
            ok = json.loads((args.dest / job["id"] / "run.json").read_text()).get("status") == "completed"
        except (OSError, ValueError):
            ok = False
        done[key] += ok
        if not ok:
            missing.append(job["id"])
    failed = sorted(path.stem for path in args.failed_dir.glob("*.json")) if args.failed_dir and args.failed_dir.is_dir() else []
    print(json.dumps({"done": sum(done.values()), "total": len(jobs),
                      "by_arm_split": {key: f"{done[key]}/{total[key]}" for key in sorted(total)},
                      "persistent_failures": failed, "not_done": len(missing)}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)
    command = sub.add_parser("inspect"); command.add_argument("--jobs", type=Path, required=True); command.set_defaults(run=inspect)
    command = sub.add_parser("split"); command.add_argument("--jobs", type=Path, required=True)
    command.add_argument("--out-dir", type=Path, required=True); command.add_argument("ips", nargs="+"); command.set_defaults(run=split)
    command = sub.add_parser("status"); command.add_argument("--jobs", type=Path, required=True)
    command.add_argument("--dest", type=Path, required=True); command.add_argument("--failed-dir", type=Path)
    command.set_defaults(run=status)
    arguments = parser.parse_args()
    arguments.run(arguments)
