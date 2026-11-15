#!/usr/bin/env python3
"""Write the fp8-KV evaluation job list for scripts/lingbot_runner_client.py."""
import json
import sys

jobs = []


def add(job_id, path, task, seed, cache, method, sink=6):
    jobs.append({"id": job_id, "path": path, "task_id": task, "seed": seed, "cache": cache, "sink": sink, "method": method})


for task in ("GC007", "GC001"):
    for path in ("turn_return", "turn_wait_return"):
        add(f"ref-bf16-c46-{task}-{path}-s42", path, task, 42, 46, "stock")
        for seed in (42, 43):
            add(f"fp8-c46-{task}-{path}-s{seed}", path, task, seed, 46, "fp8")
        add(f"fp8-c70-{task}-{path}-s42", path, task, 42, 70, "fp8")
add("fp8audit-c46-GC007-turn_wait_return-s42", "turn_wait_return", "GC007", 42, 46, "fp8audit")
# Long absence (24-chunk hold = 96 latent frames): a 46-frame window evicts the start view; 100 keeps it.
add("ref-bf16-c46-GC007-turn_long_wait_return-s42", "turn_long_wait_return", "GC007", 42, 46, "stock")
add("fp8-c46-GC007-turn_long_wait_return-s42", "turn_long_wait_return", "GC007", 42, 46, "fp8")
add("fp8-c100-GC007-turn_long_wait_return-s42", "turn_long_wait_return", "GC007", 42, 100, "fp8")
with open(sys.argv[1], "w") as handle:
    handle.write("".join(json.dumps(job) + "\n" for job in jobs))
print(len(jobs), "jobs")
