"""Rebuild the insight tail of the full-run queues with a different text prompt / chunk count.

    MODAL_PROFILE=nissan-pow python scripts/rebuild_insight_tail.py --label full1 --dry-run
    MODAL_PROFILE=nissan-pow python scripts/rebuild_insight_tail.py --label full1 --text-prompt caption --chunks 60

Pre-declared amendment (reproduction/full-run-config.json -> amendments): the insight dimension uses
`--text-prompt caption` instead of `both` for BOTH arms, chosen on 6 development cases before any insight
episode of this run existed. gc/if/outsight flags are unchanged.

What it does, per arm:
  1. reads the queue without removing items, keeps non-insight jobs in their order (this includes the
     re-queued stock retries, so they run before insight), then appends all insight jobs rebuilt with the new flags;
  2. clears and refills the queue (workers that happen to poll in the gap would exit; the script prints how many
     drain workers it expects and the client can re-spawn any);
  3. moves any ALREADY-COMPLETED insight episode that used the old prompt aside, under
     adaptive/<label>_sensitivity_both/, so it is kept as a 'both' sensitivity datum and the case is regenerated
     under the new prompt (progress record and result directory removed from the main tree).
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VOLUME = "worldsync-lingbot-results"


def rebuild_argv(job, text_prompt, chunks):
    argv = list(job["argv"])
    argv[argv.index("--text-prompt") + 1] = text_prompt
    argv[argv.index("--max-chunks") + 1] = str(chunks)
    return {**job, "argv": argv}


def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="full1")
    parser.add_argument("--text-prompt", default="caption", choices=("caption", "both", "objective"))
    parser.add_argument("--chunks", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    import modal
    out = ROOT / "outputs/adaptive" / args.label
    status = json.loads((out / "status.json").read_text())
    summary = {}
    for arm in ("stock", "budget"):
        queue = modal.Queue.from_name(f"{args.label}-{arm}")
        items = list(queue.iterate())
        keep = [job for job in items if job["split"] != "insight"]
        insight = [rebuild_argv(job, args.text_prompt, args.chunks) for job in items if job["split"] == "insight"]
        done_old = [row for row in status["arms"].get(arm, {}).get("completed", []) if row["split"] == "insight"]
        summary[arm] = {"queue_before": len(items), "non_insight_kept": len(keep), "insight_rebuilt": len(insight),
                        "completed_insight_with_old_prompt": [row["task_id"] for row in done_old]}
        if args.dry_run:
            continue
        # completed insight episodes (old prompt) -> sensitivity folder, regenerate under the new prompt
        for row in done_old:
            task = row["task_id"]
            src = f"adaptive/{args.label}/{arm}/insight-{task}-seed42__a0"
            dst = f"adaptive/{args.label}_sensitivity_both/{arm}/insight-{task}-seed42__a0"
            sh("modal", "volume", "cp", "-r", VOLUME, src, dst)
            sh("modal", "volume", "rm", "-r", VOLUME, src)
            sh("modal", "volume", "rm", VOLUME, f"adaptive/{args.label}/progress/{arm}-insight-{task}-seed42.json")
            gem = out / "gemini/episodes" / f"{arm}-insight-{task}-seed42.json"
            if gem.is_file():
                target = out / "sensitivity_both/gemini/episodes"
                target.mkdir(parents=True, exist_ok=True)
                gem.rename(target / gem.name)
            insight.append(rebuild_argv(
                __import__("worldsync.lingbot_episode_queue", fromlist=["x"]).episode_job(args.label, arm, "insight", task, 42),
                args.text_prompt, args.chunks))
        new = keep + insight
        queue.clear()
        for start in range(0, len(new), 100):
            queue.put_many(new[start:start + 100])
        summary[arm]["queue_after"] = len(new)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
