"""Job list for the frozen final run `final1` (stock vs static text director), in the Lambda worker format.

    python scripts/final_run_jobs.py      # writes reproduction/final-run-jobs.jsonl

One JSON per line: id, arm, split, task_id, seed, argv (rollout arguments without --output), output (relative to
the repo root), base, command, development (bool). A job is done when <output>/run.json has status "completed";
otherwise delete the directory and rerun (the rollout refuses a non-empty output directory).
Order: arms and dimensions interleaved so partial results stay balanced; the 20 development cases last.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABEL = "final1"
SEED = 42
SPLITS = ("gc", "if", "insight", "outsight")
COMMON = ["--player-retries", "2", "--offload-text-encoder"]
TEXT_PROMPT = {"gc": "both", "if": "both", "outsight": "both", "insight": "caption"}
MAX_CHUNKS = {"gc": 40, "if": 40, "outsight": 40, "insight": 60}
DIRECTOR = ["--director", "static", "--director-text-file", "reproduction/director-static-texts-all.json"]
DIRECTOR_SPLITS = ("if", "insight", "outsight")  # GC: the director arm is defined as identical to stock
DEVELOPMENT = {("outsight", t) for t in ("OE048", "OE037", "OE068", "OE067", "OE062", "OE043")} | \
              {("if", t) for t in ("IF012", "IF042", "IF036", "IF027", "IF011")} | \
              {("gc", t) for t in ("GC001", "GC045", "GC046")} | \
              {("insight", t) for t in ("OE008", "OE022", "OE035", "OE040", "OE063", "OE074")}


def argv_for(arm, split, task):
    argv = ["--split", split, "--task-id", task, "--seed", str(SEED), "--max-chunks", str(MAX_CHUNKS[split]),
            *COMMON, "--text-prompt", TEXT_PROMPT[split]]
    if arm == "director":
        if split not in DIRECTOR_SPLITS:
            raise ValueError("the director arm has no GC jobs: its GC episodes are the stock arm's")
        argv += DIRECTOR
    return argv


def job(arm, split, task):
    identifier = f"{arm}-{split}-{task}-seed{SEED}"
    argv = argv_for(arm, split, task)
    output = f"outputs/{LABEL}/{identifier}"
    return {"id": identifier, "arm": arm, "split": split, "task_id": task, "seed": SEED, "label": LABEL,
            "base": f"adaptive/{LABEL}/{arm}/{split}-{task}-seed{SEED}", "argv": argv, "output": output,
            "development": (split, task) in DEVELOPMENT, "needs_player": split != "insight",
            "command": "python -m worldsync.lingbot_rollout " + " ".join(argv) + f" --output {output}"}


def ordered_cases(development):
    lanes = {}
    for split in SPLITS:
        tasks = [r["task_id"] for r in json.loads((ROOT / f"external/PlayWorld/data/{split}/data.json").read_text())]
        lanes[split] = sorted((t for t in tasks if ((split, t) in DEVELOPMENT) == development),
                              key=lambda t: hashlib.sha256(f"worldsync-final1:{t}".encode()).hexdigest())
    index, out = 0, []
    while any(index < len(rows) for rows in lanes.values()):
        for split in SPLITS:
            if index < len(lanes[split]):
                out.append((split, lanes[split][index]))
        index += 1
    return out


def jobs():
    out = []
    for development in (False, True):
        for split, task in ordered_cases(development):
            out.append(job("stock", split, task))
            if split in DIRECTOR_SPLITS:
                out.append(job("director", split, task))
    return out


def main():
    rows = jobs()
    (ROOT / "reproduction/final-run-jobs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    counts = {}
    for row in rows:
        counts[(row["arm"], row["split"])] = counts.get((row["arm"], row["split"]), 0) + 1
    print(len(rows), "jobs", {f"{a}:{s}": n for (a, s), n in sorted(counts.items())},
          "development", sum(row["development"] for row in rows))


if __name__ == "__main__":
    main()
