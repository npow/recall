"""Write the fixed-path validation job list for memory v2 (step 1)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "hold": (["WAIT"] + ["RIGHT"] * 3 + ["WAIT"] * 6 + ["LEFT"] * 3 + ["WAIT"], 68),
    "translate": (["WAIT"] + ["W"] * 4 + ["S"] * 4 + ["WAIT"], 60),
    "slow360": (["WAIT"] + ["RIGHT"] * 12 + ["WAIT"], 30),
}
ARMS = {
    "stock": [],
    # Control arm: the old rule (K nearest commanded poses over a 24-frame pose-redundancy archive),
    # same K/V substitution, run through the v2 code path so it fits an 80 GB GPU.
    "oldpose": ["--memory-v2", "disk", "--memv2-select", "pose"],
    "v2": ["--memory-v2", "disk", "--memv2-select", "verified"],
    "v2content": ["--memory-v2", "disk", "--memv2-select", "content"],
}
ORDER = [("hold", "GC007"), ("translate", "GC001"), ("slow360", "GC007"), ("translate", "GC007"),
         ("hold", "GC001"), ("slow360", "GC001")]


def main():
    lines = []
    for stage, (path, task) in enumerate(ORDER):
        actions, turn = PATHS[path]
        for arm, extra in ARMS.items():
            name = f"{arm}-{path}-{task}-seed42"
            argv = ["--split", "gc", "--task-id", task, "--seed", "42", "--max-chunks", str(len(actions)),
                    "--native-fixed-controls", "--fixed-actions", *actions, "--turn-degrees-per-second", str(turn),
                    "--offload-text-encoder", *extra]
            output = f"outputs/memv2-step1/{name}"
            lines.append({"id": name, "arm": arm, "path": path, "split": "gc", "task_id": task, "seed": 42,
                          "stage": stage, "chunks": len(actions), "needs_player": False, "argv": argv, "output": output,
                          "command": "python -m worldsync.lingbot_rollout " + " ".join(argv) + f" --output {output}"})
    # Parity check of the latents-only variant against exact K/V from disk (lowest priority, untested path).
    for path, task in (("hold", "GC007"), ("hold", "GC001"), ("slow360", "GC007")):
        actions, turn = PATHS[path]
        name = f"v2recompute-{path}-{task}-seed42"
        argv = ["--split", "gc", "--task-id", task, "--seed", "42", "--max-chunks", str(len(actions)),
                "--native-fixed-controls", "--fixed-actions", *actions, "--turn-degrees-per-second", str(turn),
                "--offload-text-encoder", "--memory-v2", "recompute", "--memv2-select", "verified"]
        output = f"outputs/memv2-step1/{name}"
        lines.append({"id": name, "arm": "v2recompute", "path": path, "split": "gc", "task_id": task, "seed": 42,
                      "stage": len(ORDER), "chunks": len(actions), "needs_player": False, "argv": argv, "output": output,
                      "command": "python -m worldsync.lingbot_rollout " + " ".join(argv) + f" --output {output}"})
    target = ROOT / "reproduction/memory-v2-step1-jobs.jsonl"
    target.write_text("".join(json.dumps(line) + "\n" for line in lines))
    print(len(lines), "jobs,", sum(line["chunks"] for line in lines), "chunks ->", target)


if __name__ == "__main__":
    main()
