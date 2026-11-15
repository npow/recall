"""Drive a small Lambda Cloud fleet for adaptive LingBot episodes (run LOCALLY).

    export LAMBDA_API_KEY="$(pass show api/lambda-cloud | head -n 1)"      # only for launch/list/terminate
    python scripts/lambda_cloud.py launch --type gpu_1x_h100_pcie --region us-west-3 --ssh-key thor --name worldsync-optb-2
    python scripts/lambda_fleet.py setup IP [IP ...]         # push code + data, build env, fetch + verify weights
    python scripts/lambda_fleet.py canary IP                 # fixed 8-chunk stock canary (parity check)
    python scripts/lambda_fleet.py tunnel IP [IP ...]        # reverse tunnel to the local player proxy (keep running)
    python scripts/lambda_fleet.py jobs --label optb --arm stock --seed 42 --all --out outputs/lambda/optb/jobs.jsonl
    python scripts/lambda_fleet.py start --jobs outputs/lambda/optb/jobs.jsonl IP [IP ...]   # one shard per IP
    python scripts/lambda_fleet.py pull --label optb IP [IP ...]   # rsync results to outputs/lambda/<label>/
    python scripts/lambda_fleet.py status IP [IP ...]
    python scripts/lambda_cloud.py terminate INSTANCE_ID ...

Player calls go to the Anthropic-compatible proxy on THIS machine (127.0.0.1:8317) through an SSH reverse
tunnel; the worker runs with ANTHROPIC_BASE_URL=http://127.0.0.1:8317 and the proxy client key, which is sent
through the SSH session's stdin only; it is not written to the instance's disk, the repository or any log.
Keep the fleet at <= 8 instances: each worker makes one player request at a time (subscription rate limits);
the Anthropic SDK client in the rollout retries 429s with backoff, and the queue retries transient failures.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SHARED = Path(os.environ.get("WORLDSYNC_SHARED_ROOT", "/home/npow/code/iteratehack"))
KEY = os.path.expanduser(os.environ.get("LAMBDA_SSH_KEY", "~/.ssh/id_rsa"))
SSH = ["ssh", "-i", KEY, "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30"]
RSYNC_SSH = " ".join(SSH)
PYTHON = ".venv-lambda/bin/python"


def ssh(ip, script, stdin_prefix="", check=True, capture=False):
    """Run a bash script on the instance; the script travels on stdin, so nothing secret hits argv."""
    return subprocess.run([*SSH, f"ubuntu@{ip}", "bash -s"], input=stdin_prefix + script, text=True, check=check,
                          capture_output=capture)


def rsync(source, destination, *flags):
    subprocess.run(["rsync", "-az", "--partial", "-e", RSYNC_SSH, *flags, source, destination], check=True)


def push(ip, source=ROOT, remote="work"):
    """rsync a checkout (any worktree / branch; the repository is private, so nothing is cloned) to ~/<remote>.

    remote="work" is the base tree (origin/main + environment + weights + data). Any other name is a
    second code tree for a different branch; it shares the base tree's environment, weights and data by symlink.
    """
    source = Path(source)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=source, capture_output=True, text=True).stdout.strip()
    ssh(ip, f"mkdir -p ~/{remote}/reproduction ~/work/external/PlayWorld\n")
    rsync(f"{source}/worldsync", f"ubuntu@{ip}:{remote}/", "--delete", "--exclude", "__pycache__")
    if source != ROOT:
        rsync(f"{source}/scripts", f"ubuntu@{ip}:{remote}/", "--exclude", "__pycache__")
    rsync(f"{ROOT}/scripts", f"ubuntu@{ip}:{remote}/", "--exclude", "__pycache__")  # fleet scripts always from here
    rsync(f"{source}/reproduction/", f"ubuntu@{ip}:{remote}/reproduction/", "--include", "*.json", "--exclude", "*",
          "--max-size", "5m")  # top-level JSON only (pins, state, configs, dev texts)
    rsync(f"{source}/pyproject.toml", f"ubuntu@{ip}:{remote}/")
    if remote == "work":
        rsync(f"{SHARED}/external/PlayWorld/data", f"ubuntu@{ip}:work/external/PlayWorld/", "--exclude", "*.mp4")
    else:
        ssh(ip, f"cd ~/{remote} && ln -sfn ~/work/external external && ln -sfn ~/work/.venv-lambda .venv-lambda && mkdir -p results outputs\n")
    ssh(ip, f"echo '{commit}{' +local-changes' if dirty else ''}' > ~/{remote}/COMMIT\n")
    return commit


def push_command(args):
    for ip in args.ips:
        print(json.dumps({"ip": ip, "remote": args.remote, "pushed_commit": push(ip, args.source, args.remote)}))


def setup(args):
    for ip in args.ips:
        print(json.dumps({"ip": ip, "pushed_commit": push(ip)}), flush=True)
        ssh(ip, (ROOT / "scripts/lambda_setup.sh").read_text())
        ssh(ip, f"cd ~/work && HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DOWNLOAD_TIMEOUT=180 {PYTHON} scripts/lambda_fetch_weights.py\n")


PROXY_PORT = int(os.environ.get("WORLDSYNC_PROXY_PORT", "8317"))
PROXY_KEY_FILE = Path(os.environ.get("WORLDSYNC_PROXY_KEY_FILE", "/home/npow/.local/opt/cliproxyapi/client-key.txt"))


def secret_prefix():
    """Player credentials for the instance: the local Anthropic-compatible proxy, reached through a reverse tunnel.

    The project's own Anthropic key is NOT sent anywhere. The proxy client key goes over the SSH session's
    stdin into the worker's environment only.
    """
    key = PROXY_KEY_FILE.read_text().strip()
    if not key:
        sys.exit(f"empty proxy client key file: {PROXY_KEY_FILE}")
    return f"export ANTHROPIC_API_KEY='{key}'\nexport ANTHROPIC_BASE_URL='http://127.0.0.1:{PROXY_PORT}'\n"


def tunnel(args):
    """Keep a reverse tunnel instance:PROXY_PORT -> this machine's proxy alive (restart loop, detached)."""
    logs = ROOT / "outputs/lambda"
    logs.mkdir(parents=True, exist_ok=True)
    for ip in args.ips:
        command = " ".join([*SSH, "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveCountMax=3",
                            "-R", f"{PROXY_PORT}:127.0.0.1:{PROXY_PORT}", f"ubuntu@{ip}"])
        loop = f"while true; do {command}; sleep 5; done"
        with open(logs / f"tunnel-{ip}.log", "ab") as log:
            process = subprocess.Popen(["bash", "-c", loop], stdout=log, stderr=log, start_new_session=True)
        print(json.dumps({"ip": ip, "tunnel_pid": process.pid, "stop": f"kill -- -{process.pid}"}))


def canary(args):
    for ip in args.ips:
        ssh(ip, f"cd ~/work && {PYTHON} -u scripts/lambda_worker.py --canary-only 2>&1 | grep -E '^\\{{' \n")


def jobs(args):
    from worldsync import lingbot_episode_queue as q
    data = SHARED / "external/PlayWorld/data"
    rows = []
    for split in ("gc", "if", "outsight", "insight"):
        tasks = [record["task_id"] for record in json.loads((data / split / "data.json").read_text())]
        if args.cases:
            tasks = [task for task in tasks if task in set(args.cases)]
        elif not args.all:
            tasks = []
        rows.append([q.episode_job(args.label, args.arm, split, task, args.seed, extra=args.extra) for task in tasks])
    ordered = []  # interleave the three player-driven splits; the long stationary insight episodes go last
    for index in range(max(len(group) for group in rows[:3])):
        ordered.extend(group[index] for group in rows[:3] if index < len(group))
    ordered.extend(rows[3])
    for job in ordered:
        for flag, value in args.replace or []:  # e.g. --replace --text-prompt caption
            if flag in job["argv"]:
                job["argv"][job["argv"].index(flag) + 1] = value
        if job["split"] == "insight" and args.insight_text_prompt:
            job["argv"][job["argv"].index("--text-prompt") + 1] = args.insight_text_prompt
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(job) + "\n" for job in ordered))
    print(json.dumps({"jobs": len(ordered), "out": str(args.out), "example_argv": ordered[0]["argv"] if ordered else None}))


def start(args):
    prefix = "" if args.no_player else secret_prefix()
    shards = args.shards or len(args.ips)  # --shards/--first-shard let instances join one job list at different times
    for shard, ip in enumerate(args.ips, start=args.first_shard):
        rsync(str(args.jobs), f"ubuntu@{ip}:{args.remote}/jobs.jsonl")
        flags = f"--jobs jobs.jsonl --shard {shard} --shards {shards}" + (" --canary" if args.canary else "")
        if args.plain:
            flags += f" --plain {args.plain} --plain-retries {args.plain_retries}"
        # A fresh tmux server inherits the exported key from this shell, so it never appears in argv or on disk.
        # The worker is restarted (fresh model) up to 5 times if it exits non-zero, e.g. after a fatal CUDA error.
        if getattr(args, "gpus", 1) > 1:
            # Multi-GPU box: one worker per GPU, the instance's list split by line index % gpus, logs worker-<gpu>.log.
            script = f"cd ~/{args.remote} && (tmux kill-server 2>/dev/null || true) && sleep 1\n"
            for gpu in range(args.gpus):
                gpu_flags = flags.replace(f"--shard {shard} --shards {shards}", f"--shard {gpu} --shards {args.gpus}")
                loop = (f"for n in 1 2 3 4 5; do CUDA_VISIBLE_DEVICES={gpu} {PYTHON} -u scripts/lambda_worker.py {gpu_flags} "
                        f">> worker-{gpu}.log 2>&1 && break; echo worker-exited-nonzero >> worker-{gpu}.log; sleep 5; done")
                script += f": > worker-{gpu}.log && tmux new-session -d -s worker{gpu} '{loop}'\n"
            ssh(ip, script + "echo started\n", stdin_prefix=prefix)
            continue
        loop = (f"for n in 1 2 3 4 5; do {PYTHON} -u scripts/lambda_worker.py {flags} >> worker.log 2>&1 && break; "
                "echo worker-exited-nonzero >> worker.log; sleep 5; done")
        ssh(ip, f"cd ~/{args.remote} && (tmux kill-server 2>/dev/null || true) && sleep 1 && : > worker.log && "
                f"tmux new-session -d -s worker '{loop}' && echo started\n", stdin_prefix=prefix)


def pull_dev(args):
    """Dev results: results/dev/<label>/<id>/{rollout.mp4,run.json} -> <dest>/<id>/ (repeat to pick up new ones)."""
    args.dest.mkdir(parents=True, exist_ok=True)
    for ip in args.ips:
        try:
            rsync(f"ubuntu@{ip}:{args.remote}/results/dev/{args.label}/", f"{args.dest}/", "--include", "*/",
                  "--include", "rollout.mp4", "--include", "run.json", "--exclude", "*")
            # kept failed attempts, and persistent-failure markers (raw: may include "moved to another instance" markers)
            for extra, local in (("attempts", "attempts"), ("failed", "failed-raw")):
                side = args.dest.parent / f"{args.dest.name}-{local}"
                side.mkdir(parents=True, exist_ok=True)
                subprocess.run(["rsync", "-az", "-e", RSYNC_SSH, "--include", "*/", "--include", "run.json",
                                "--include", "attempt-error.txt", "--include", "*.json", "--exclude", "*",
                                f"ubuntu@{ip}:{args.remote}/results/dev/{args.label}-{extra}/", f"{side}/"],
                               check=False, capture_output=True)
            raw, real = args.dest.parent / f"{args.dest.name}-failed-raw", args.dest.parent / f"{args.dest.name}-failed"
            real.mkdir(parents=True, exist_ok=True)
            for marker in raw.glob("*.json"):  # only genuine persistent failures reach <dest>-failed
                if "stolen_by" not in marker.read_text() and not (args.dest / marker.stem / "run.json").exists():
                    (real / marker.name).write_text(marker.read_text())
        except subprocess.CalledProcessError as failure:  # an instance that is not up yet must not stop the others
            print(json.dumps({"ip": ip, "pull_error": str(failure)[:120]}))
    done = sum(1 for path in args.dest.glob("*/run.json"))
    print(json.dumps({"label": args.label, "episodes_with_run_json": done, "dest": str(args.dest)}))


def pull(args):
    target = ROOT / "outputs/lambda" / args.label
    target.mkdir(parents=True, exist_ok=True)
    for ip in args.ips:
        rsync(f"ubuntu@{ip}:work/results/adaptive/{args.label}/", f"{target}/", "--include", "*/", "--include", "rollout.mp4",
              "--include", "run.json", "--include", "progress/*.json", "--include", "failed/*.json", "--exclude", "*")
        rsync(f"ubuntu@{ip}:work/worker.log", f"{target}/worker-{ip}.log")
    done = len(list((target / "progress").glob("*.json"))) if (target / "progress").is_dir() else 0
    print(json.dumps({"label": args.label, "progress_records": done, "target": str(target)}))


def status(args):
    for ip in args.ips:
        out = ssh(ip, f"cd ~/{args.remote}; ls -d results/dev/*/* 2>/dev/null | wc -l; ls results/adaptive/*/progress 2>/dev/null | wc -l; ls results/adaptive/*/failed 2>/dev/null | wc -l; "
                      "tmux has-session -t worker 2>/dev/null && echo running || echo stopped; "
                      "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; grep -E '^\\{' worker.log | tail -n 2\n",
                  capture=True, check=False)
        print(f"== {ip}\n{out.stdout.strip()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default="work", help="code tree on the instance (~/<remote>); 'work' is the base tree")
    sub = parser.add_subparsers(required=True)
    command = sub.add_parser("push"); command.add_argument("--source", type=Path, default=ROOT)
    command.add_argument("ips", nargs="+"); command.set_defaults(run=push_command)
    command = sub.add_parser("pull-dev"); command.add_argument("--label", required=True)
    command.add_argument("--dest", type=Path, required=True); command.add_argument("ips", nargs="+")
    command.set_defaults(run=pull_dev)
    for name, function in (("setup", setup), ("canary", canary), ("status", status), ("tunnel", tunnel)):
        command = sub.add_parser(name); command.add_argument("ips", nargs="+"); command.set_defaults(run=function)
    command = sub.add_parser("jobs")
    command.add_argument("--label", required=True); command.add_argument("--arm", default="stock")
    command.add_argument("--seed", type=int, default=42); command.add_argument("--all", action="store_true")
    command.add_argument("--cases", nargs="*"); command.add_argument("--extra", nargs=argparse.REMAINDER)
    command.add_argument("--replace", nargs=2, action="append", metavar=("FLAG", "VALUE"))
    command.add_argument("--insight-text-prompt", default="caption", help="full-run amendment: insight uses the caption only")
    command.add_argument("--out", type=Path, required=True); command.set_defaults(run=jobs)
    command = sub.add_parser("start")
    command.add_argument("--jobs", type=Path, required=True); command.add_argument("--canary", action="store_true")
    command.add_argument("--plain", metavar="LABEL", help="dev jobs: results/dev/LABEL/<id>/")
    command.add_argument("--no-player", action="store_true", help="fixed-action jobs: send no player credentials")
    command.add_argument("--shards", type=int); command.add_argument("--first-shard", type=int, default=0)
    command.add_argument("--plain-retries", type=int, default=0, help="extra attempts per job (failed attempts are kept)")
    command.add_argument("--gpus", type=int, default=1, help="multi-GPU box: one worker per GPU on each IP")
    command.add_argument("ips", nargs="+"); command.set_defaults(run=start)
    command = sub.add_parser("pull"); command.add_argument("--label", required=True)
    command.add_argument("ips", nargs="+"); command.set_defaults(run=pull)
    arguments = parser.parse_args()
    arguments.run(arguments)
