"""List reference-plan durations for the development cohort (never touches held-out cases)."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
split = json.loads((ROOT / "reproduction/episode-split.json").read_text())
development = {(c["split"], c["task_id"]) for c in split["development"]}
rows = []
for name in ("gc", "if", "insight", "outsight"):
    for task in json.loads((ROOT / f"external/PlayWorld/data/{name}/data.json").read_text()):
        if (name, task["task_id"]) not in development:
            continue
        steps = task.get("action_sequence_steps") or []
        total = 0
        for step in steps:
            match = re.search(r"(\d+)ms", step)
            total += int(match.group(1)) if match else 0
        rows.append((total / 1000, name, task["task_id"], steps, task["prompt"][:90]))
rows.sort(reverse=True)
limit = int(sys.argv[1]) if len(sys.argv) > 1 else 25
for seconds, name, task_id, steps, prompt in rows[:limit]:
    print(f"{seconds:5.1f}s {name:8s} {task_id:6s} {steps} | {prompt}")
print(len(rows), "development cases")
