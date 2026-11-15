#!/usr/bin/env python3
"""Submit jobs.jsonl to the deployed persistent LingBot runner (Modal profile nissan-pow).

    MODAL_PROFILE=nissan-pow modal deploy scripts/modal_lingbot_runner.py     # once
    MODAL_PROFILE=nissan-pow python scripts/lingbot_runner_client.py jobs.jsonl --gpu H100 --containers 2

Each line of jobs.jsonl is a job (see worldsync/lingbot_runner_core.py). A `method` field is
expanded client-side into rollout arguments:

    stock | pose | pose-native | kv<N> | budget<K> | budget<K>native | reanchor | decoprune
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

METHOD_ARGS = {
    "stock": [],
    "pose": ["--pose-retrieval"],
    "pose-native": ["--pose-retrieval", "--retrieval-positions", "native"],
    "reanchor": ["--reanchor"],
    "decoprune": ["--decoprune"],
}


def method_args(method):
    if method in METHOD_ARGS:
        return list(METHOD_ARGS[method])
    if method.startswith("kv") and method[2:].isdigit():
        return ["--kv-retention", "--kv-retained-frames", method[2:]]
    if method.startswith("budget"):
        rest = method[len("budget"):]
        rope = "native" if rest.endswith("native") else "virtual"
        digits = rest.removesuffix("native").removesuffix("virtual")
        if digits.isdigit():
            return ["--budget-memory", "--budget-retrieved", digits, "--budget-archive", "24", "--budget-rope", rope]
    raise ValueError(f"unknown method {method!r}")


def expand(job):
    job = dict(job)
    method = job.pop("method", None)
    if method:
        job["extra"] = method_args(method) + list(job.get("extra", []))
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--gpu", choices=["H100", "H200"], default="H200")
    parser.add_argument("--containers", type=int, default=2, help="concurrent jobs = GPU containers used (capacity is scarce)")
    parser.add_argument("--app", default="worldsync-lingbot-runner")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    import modal
    runner = modal.Cls.from_name(args.app, "RunnerH100" if args.gpu == "H100" else "RunnerH200")()
    entries = [expand(json.loads(line)) for line in args.jobs.read_text().splitlines() if line.strip()]

    def one(job):
        sent = time.monotonic()
        result = runner.run.remote(job)
        result["client_wall_s"] = time.monotonic() - sent
        print(json.dumps({key: result.get(key) for key in ("id", "status", "error", "job_seconds", "client_wall_s")}), flush=True)
        return result

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.containers) as pool:
        outcomes = list(pool.map(one, entries))
    out = args.out or Path(f"runner-results-{int(time.time())}.json")
    out.write_text(json.dumps({"gpu": args.gpu, "containers": args.containers,
                               "total_wall_s": time.monotonic() - started, "results": outcomes}, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
