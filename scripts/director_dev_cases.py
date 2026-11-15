"""Pick the director development cases (never by outcome) and print their stock full-run trajectories.

Selection: development split of reproduction/episode-split.json, ordered by a salted hash of the case id;
6 out-of-sight, 5 IF, 3 GC. Insight uses the six cases already declared as insight development cases in
reproduction/full-run-config.json (OE008, OE022, OE035, OE040, OE063, OE074).
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SALT = "worldsync-director-dev-v1"
COUNTS = {"outsight": 6, "if": 5, "gc": 3}
INSIGHT_DEV = ["OE008", "OE022", "OE035", "OE040", "OE063", "OE074"]


def select():
    split = json.loads((ROOT / "reproduction/episode-split.json").read_text())
    development = {(row["split"], row["task_id"]) for row in split["development"]}
    chosen = []
    for name, count in COUNTS.items():
        rows = sorted((case for case in development if case[0] == name and case[1] != "GC013"),
                      key=lambda case: hashlib.sha256(f"{SALT}:{case[1]}".encode()).hexdigest())
        chosen += rows[:count]
    chosen += [("insight", task) for task in INSIGHT_DEV]
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-run", type=Path, required=True, help="outputs/adaptive/full1 of the full run")
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    rows = []
    for split, task in select():
        record = next(r for r in json.loads((ROOT / f"external/PlayWorld/data/{split}/data.json").read_text()) if r["task_id"] == task)
        episode = args.full_run / "episodes" / f"stock-{split}-{task}-seed42"
        run = json.loads((episode / "run.json").read_text()) if (episode / "run.json").is_file() else {}
        judged = args.full_run / "gemini/episodes" / f"stock-{split}-{task}-seed42.json"
        score = (json.loads(judged.read_text()).get("scores") or {}).get("official") if judged.is_file() else None
        actions = [a.get("action") for a in run.get("actions", []) if a.get("operation") != "END" and a.get("action") not in (None, "NONE")]
        rows.append({"split": split, "task_id": task, "caption": record["image_caption"], "objective": record["prompt"],
                     "plan": record.get("action_sequence_steps"), "stock_actions": actions, "stock_chunks": len(run.get("chunks", [])),
                     "stock_termination": run.get("termination"), "stock_text_prompt_mode": run.get("text_prompt_mode"),
                     "stock_single_call_score": score, "stock_video": str(episode / "rollout.mp4")})
    for row in rows:
        print(row["split"], row["task_id"], "| chunks", row["stock_chunks"], "| score", row["stock_single_call_score"], "|", row["stock_text_prompt_mode"],
              "|", " ".join(row["stock_actions"]))
        print("   objective:", row["objective"][:150], "| plan:", row["plan"])
        print("   caption:", row["caption"])
    if args.write:
        args.write.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
