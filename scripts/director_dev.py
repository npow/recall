"""Director development experiment: fixed-action replays of the stock full-run trajectories, text varied.

Arms (all stock weights, 18/6 window unless stated, seed 42, same commanded actions as the existing stock episode):
  replay   same text as the stock episode (bit-exactness check of the replay harness; a few cases only)
  dstatic  --director static with preset texts            (all 20 dev cases)
  dsched   --director unseen with preset scheduled texts  (insight + out-of-sight)
  sink2    stock text, --sink-size 2 (controlled ablation) (insight + out-of-sight)

    python scripts/director_dev.py jobs      # print the job list (argv per episode)
    MODAL_PROFILE=nissan-pow python scripts/director_dev.py submit --containers 4
    MODAL_PROFILE=nissan-pow python scripts/director_dev.py status
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
APP_NAME = "worldsync-lingbot-director"
LABEL = "director-dev2"
QUEUE = f"{LABEL}-jobs"
TEXTS = "/work/reproduction/director-dev-texts.json"
LLM_TEXTS = "reproduction/director-dev-texts-llm.json"
PRESET_TEXTS = "reproduction/director-dev-texts.json"
REPLAY_CHECK = {("outsight", "OE048"), ("if", "IF042"), ("gc", "GC045")}


def cases():
    return json.loads((ROOT / "outputs/director-dev-cases.json").read_text())


def argv_for(arm, row, texts=TEXTS):
    split, task = row["split"], row["task_id"]
    actions = row["stock_actions"]
    # --max-chunks must equal the stock run's value: upstream draws the noise for the whole sequence at once, so a
    # different frame budget changes every chunk's noise. The episode still ends when the fixed actions run out.
    argv = ["--split", split, "--task-id", task, "--seed", "42", "--max-chunks", "60" if split == "insight" else "40",
            "--fixed-actions", *actions, "--player-retries", "2",
            "--text-prompt", row["stock_text_prompt_mode"], "--offload-text-encoder"]
    if arm == "dstatic":
        argv += ["--director", "static", "--director-text-file", texts]
    elif arm == "dsched":
        argv += ["--director", "unseen", "--director-text-file", texts]
    elif arm == "lstatic":
        argv += ["--director", "static", "--director-text-file", LLM_TEXTS]
    elif arm == "lsched":
        argv += ["--director", "unseen", "--director-text-file", LLM_TEXTS]
    elif arm == "nag":
        argv += ["--nag"]
    elif arm == "sink2":
        argv += ["--sink-size", "2"]
    elif arm != "replay":
        raise ValueError(arm)
    return argv


def jobs(texts=TEXTS):
    out = []
    for row in cases():
        key = (row["split"], row["task_id"])
        arms = ["dstatic"]
        if row["split"] in ("insight", "outsight"):
            arms += ["dsched", "sink2"]
        if key in REPLAY_CHECK:
            arms.append("replay")
        for arm in arms:
            identifier = f"{arm}-{row['split']}-{row['task_id']}-seed42"
            out.append({"id": identifier, "arm": arm, "split": row["split"], "task_id": row["task_id"], "seed": 42,
                        "label": LABEL, "base": f"adaptive/{LABEL}/{arm}/{row['split']}-{row['task_id']}-seed42",
                        "argv": argv_for(arm, row, texts)})
    # short episodes first so results arrive early; the 60-chunk insight episodes go last
    return sorted(out, key=lambda job: (job["split"] == "insight", job["arm"] != "replay"))


def lambda_jobs():
    """Self-contained job list for plain GPU boxes, most informative first. Every case gets a `replay` control
    (stock text) on the same hardware, because outputs are only bit-exact within one GPU model."""
    rows = cases()
    by = lambda split: [row for row in rows if row["split"] == split]  # noqa: E731
    stages = [(by("if"), ["replay", "lstatic"]),
              (by("insight"), ["replay", "lstatic"]),
              (by("outsight"), ["replay", "lstatic", "lsched"]),
              (by("gc"), ["replay", "lstatic"]),
              (by("if") + by("outsight"), ["nag"]),
              (by("outsight"), ["sink2"]),
              (by("insight"), ["lsched"]),
              (by("insight"), ["sink2"]),
              (by("insight"), ["nag"]),
              (by("if") + by("outsight") + by("gc"), ["dstatic"]),
              (by("outsight"), ["dsched"]),
              (by("insight"), ["dstatic"]),
              (by("insight"), ["dsched"])]
    out = []
    for stage, (group, arms) in enumerate(stages):
        for row in group:
            for arm in arms:
                identifier = f"{arm}-{row['split']}-{row['task_id']}-seed42"
                argv = argv_for(arm, row, PRESET_TEXTS)
                output = f"outputs/director-lambda/{identifier}"
                out.append({"id": identifier, "arm": arm, "split": row["split"], "task_id": row["task_id"], "seed": 42,
                            "stage": stage, "label": "director-lambda", "base": f"adaptive/director-lambda/{arm}/{row['split']}-{row['task_id']}-seed42",
                            "argv": argv, "output": output, "chunks": len(row["stock_actions"]),
                            "command": "python -m worldsync.lingbot_rollout " + " ".join(argv) + f" --output {output}"})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["jobs", "submit", "status", "jsonl"])
    parser.add_argument("--containers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "jobs":
        for job in jobs():
            print(job["id"], " ".join(job["argv"])[:200])
        print(len(jobs()), "jobs")
        return
    if args.command == "jsonl":
        rows = lambda_jobs()
        text = "".join(json.dumps(row) + "\n" for row in rows)
        for target in (ROOT / "reproduction/director-lambda-jobs.jsonl", ROOT / "outputs/director-lambda-jobs.jsonl"):
            target.write_text(text)
        print(len(rows), "jobs;", sum(row["chunks"] for row in rows), "chunks")
        return
    import modal
    if args.command == "submit":
        queue = modal.Queue.from_name(QUEUE, create_if_missing=True)
        queue.clear()
        queue.put_many(jobs())
        worker = modal.Cls.from_name(APP_NAME, "AdaptiveH200")()
        calls = [worker.drain.spawn(QUEUE, LABEL, None, 2, False).object_id for _ in range(args.containers)]
        (ROOT / "outputs/director-dev-calls.json").write_text(json.dumps(calls) + "\n")
        print(f"queued {len(jobs())} jobs, spawned {len(calls)} workers")
    else:
        report = modal.Function.from_name(APP_NAME, "status").remote(LABEL)
        (ROOT / "outputs/director-dev-status.json").write_text(json.dumps(report, indent=1) + "\n")
        for arm, rows in sorted(report["arms"].items()):
            print(arm, "completed", rows["completed_count"], "failed", len(rows["failed"]))
            for failure in rows["failed"]:
                print("   FAILED", failure["split"], failure["task_id"], str(failure.get("attempts"))[:300])


if __name__ == "__main__":
    main()
