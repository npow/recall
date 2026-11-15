"""Write the Lambda job list for the memory-storage ablation (fixed native paths, no player)."""
import json
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "turn_hold_return": ["WAIT"] + ["RIGHT"] * 3 + ["WAIT"] * 6 + ["LEFT"] * 3 + ["WAIT"],
    "turn_return": ["WAIT"] + ["RIGHT"] * 3 + ["LEFT"] * 3 + ["WAIT"],
}
# (name, store codec or None for stock, layer subset)
VARIANTS = [
    ("stock", None, "all"), ("exact", "exact", "all"),
    ("fp8", "fp8", "all"), ("int8", "int8", "all"),
    ("L-even", "exact", "even"), ("L-every4", "exact", "every4"), ("L-first10", "exact", "first10"),
    ("L-last10", "exact", "last10"), ("L-first20", "exact", "first20"), ("L-last20", "exact", "last20"),
    ("L-mid20", "exact", "mid20"),
    ("pool2", "pool2", "all"), ("pool4", "pool4", "all"),
    ("top25", "top25", "all"), ("top10", "top10", "all"), ("rand25", "rand25", "all"),
    ("lr64", "lr64", "all"), ("lr256", "lr256", "all"),
    ("pool2fp8", "pool2+fp8", "all"), ("pool2fp8-even", "pool2+fp8", "even"),
]
SHORT_PATH_VARIANTS = {"stock", "exact", "fp8", "L-even", "pool2", "top25", "lr64", "pool2fp8"}


def job(variant, store, layers, task, path, seed, stage, retrieved=10, prefix="ms"):
    actions = PATHS[path]
    identifier = f"{prefix}-{variant}-{task}-{path}-seed{seed}"
    argv = ["--split", "gc", "--task-id", task, "--seed", str(seed), "--max-chunks", str(len(actions)),
            "--native-fixed-controls", "--fixed-actions", *actions, "--turn-degrees-per-second", "68",
            "--offload-text-encoder"]
    if store is not None:
        argv += ["--budget-memory", "--budget-rope", "native", "--budget-retrieved", str(retrieved),
                 "--budget-store", store, "--budget-layers", layers]
    output = f"outputs/memstore/{identifier}"
    return {"id": identifier, "arm": variant, "split": "gc", "task_id": task, "path": path, "seed": seed,
            "stage": stage, "needs_player": False, "argv": argv, "output": output, "chunks": len(actions),
            "command": "python -m worldsync.lingbot_rollout " + shlex.join(argv) + f" --output {output}"}


def build(retrieved=10, prefix="ms"):
    jobs = []
    for stage, (task, path) in enumerate([("GC007", "turn_hold_return"), ("GC001", "turn_hold_return")]):
        jobs += [job(name, store, layers, task, path, 42, stage, retrieved, prefix)
                 for name, store, layers in VARIANTS]
    jobs += [job(name, store, layers, "GC007", "turn_return", 42, 2, retrieved, prefix)
             for name, store, layers in VARIANTS if name in SHORT_PATH_VARIANTS]
    return jobs


if __name__ == "__main__":
    for retrieved, prefix, name in ((10, "ms", "memory-storage-jobs.jsonl"), (6, "ms6", "memory-storage-jobs-k6.jsonl")):
        jobs = build(retrieved, prefix)
        (ROOT / "reproduction" / name).write_text("".join(json.dumps(item) + "\n" for item in jobs))
        print(name, len(jobs), "jobs", sum(item["chunks"] for item in jobs), "chunks")
