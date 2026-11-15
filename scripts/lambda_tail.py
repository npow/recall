"""Move not-yet-started jobs from the busiest instances to instances that have finished their list (run LOCALLY).

    python scripts/lambda_tail.py --remote final --label final1 --lists outputs/lambda/final1 [--apply] IP [IP ...]

Only idle instances (worker finished its list) receive jobs, so no running job is interrupted. Jobs are taken
from the END of the busiest instance's remaining list (its lowest-priority jobs), leaving its next two jobs alone.
The donor is told to skip a moved job with a marker `results/dev/<label>-failed/<id>.json` containing
`stolen_by`; lambda_fleet.py pull-dev keeps such markers out of the persistent-failure folder.
Without --apply it only prints the plan.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lambda_final
import lambda_fleet as fleet


def state(ip, remote, label):
    script = (f"cd ~/{remote}; for d in results/dev/{label}/*/; do grep -q '\"status\": \"completed\"' $d/run.json 2>/dev/null && echo DONE $(basename $d); done; "
              f"ls results/dev/{label}-failed 2>/dev/null | sed 's/\\.json$//; s/^/FAILED /'; "
              f"echo INFLIGHT $(ls -t outputs/dev/{label} 2>/dev/null | head -n 1); "
              "grep -q '\"event\": \"summary\"' worker.log && echo IDLE; tmux has-session -t worker 2>/dev/null || echo IDLE\n")
    out = fleet.ssh(ip, script, capture=True, check=False).stdout
    done, failed, inflight, idle = set(), set(), None, False
    for line in out.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "DONE" and len(parts) == 2:
            done.add(parts[1])
        elif parts[0] == "FAILED" and len(parts) == 2:
            failed.add(parts[1])
        elif parts[0] == "INFLIGHT" and len(parts) == 2:
            inflight = parts[1]
        elif parts[0] == "IDLE":
            idle = True
    return done, failed, None if idle else inflight, idle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--lists", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("ips", nargs="+")
    args = parser.parse_args()
    remaining, idle, all_done = {}, [], set()
    for ip in args.ips:
        done, failed, inflight, is_idle = state(ip, args.remote, args.label)
        all_done |= done
        files = sorted(args.lists.glob(f"jobs-{ip}*.jsonl"))  # original list plus any later tail lists
        jobs = [job for path in files for job in lambda_final.load(path)]
        remaining[ip] = [job for job in jobs if job["id"] not in done and job["id"] not in failed and job["id"] != inflight]
        if is_idle:
            idle.append(ip)
            remaining[ip] = []
    for ip in remaining:  # a job finished elsewhere (moved earlier) is not remaining
        remaining[ip] = [job for job in remaining[ip] if job["id"] not in all_done]
    load = {ip: sum(lambda_final.seconds(job) for job in rows) for ip, rows in remaining.items()}
    moves = {ip: [] for ip in idle}
    stolen = {}
    while idle:
        donor = max(args.ips, key=lambda ip: load[ip])
        receiver = min(idle, key=lambda ip: load[ip])
        if len(remaining[donor]) <= 2 or load[donor] - load[receiver] < 400:
            break
        job = remaining[donor].pop()
        cost = lambda_final.seconds(job)
        load[donor] -= cost
        load[receiver] += cost
        moves[receiver].insert(0, job)  # popped from the end, so re-insert in the original priority order
        stolen.setdefault(donor, []).append((job["id"], receiver))
    plan = {"idle": idle, "remaining_minutes": {ip: round(load[ip] / 60) for ip in args.ips},
            "moves": {ip: [job["id"] for job in rows] for ip, rows in moves.items()}}
    print(json.dumps(plan, indent=1))
    if not args.apply:
        return
    for donor, rows in stolen.items():
        script = f"mkdir -p ~/{args.remote}/results/dev/{args.label}-failed && cd ~/{args.remote}/results/dev/{args.label}-failed\n"
        for identifier, receiver in rows:
            script += f"echo '{json.dumps({'id': identifier, 'stolen_by': receiver})}' > {identifier}.json\n"
        fleet.ssh(donor, script)
    for index, (receiver, rows) in enumerate(moves.items()):
        if not rows:
            continue
        number = len(list(args.lists.glob(f"jobs-{receiver}*.jsonl")))
        path = args.lists / f"jobs-{receiver}-tail{number}.jsonl"
        path.write_text("".join(json.dumps(job) + "\n" for job in rows))
        start = argparse.Namespace(no_player=False, ips=[receiver], remote=args.remote, jobs=path, shards=1, first_shard=0,
                                   canary=False, plain=args.label, plain_retries=args.retries)
        fleet.start(start)


if __name__ == "__main__":
    main()
