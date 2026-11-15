"""Verify the full-run queues and workers (read-only).

    MODAL_PROFILE=nissan-pow python scripts/verify_full_run_queues.py --label full1

Checks, per arm: every job is either completed, queued exactly once, or (at most `workers`) in flight; no duplicates;
no queued job already completed; insight jobs carry --text-prompt caption / --max-chunks 60; non-insight jobs carry
--text-prompt both; and whether each spawned drain worker is still running.
"""
import argparse
import collections
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def flag(job, name):
    return job["argv"][job["argv"].index(name) + 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="full1")
    parser.add_argument("--insight-prompt", default="caption")
    parser.add_argument("--insight-chunks", default="60")
    args = parser.parse_args()
    import modal
    from adaptive_runner_client import all_cases
    out = ROOT / "outputs/adaptive" / args.label
    status = json.loads((out / "status.json").read_text())
    cases = all_cases(ROOT / "external/PlayWorld/data")
    for arm in ("stock", "budget"):
        items = list(modal.Queue.from_name(f"{args.label}-{arm}").iterate())
        ids = [job["id"] for job in items]
        dups = [k for k, c in collections.Counter(ids).items() if c > 1]
        done = {f"{arm}-{r['split']}-{r['task_id']}-seed42" for r in status["arms"][arm]["completed"]}
        failed = [f"{arm}-{r['split']}-{r['task_id']}-seed42" for r in status["arms"][arm]["failed"]]
        every = {f"{arm}-{s}-{t}-seed42" for s, t in cases}
        in_flight_or_missing = sorted(every - set(ids) - done)
        insight = [job for job in items if job["split"] == "insight"]
        bad_insight = [job["id"] for job in insight if flag(job, "--text-prompt") != args.insight_prompt
                       or flag(job, "--max-chunks") != args.insight_chunks]
        bad_other = [job["id"] for job in items if job["split"] != "insight" and flag(job, "--text-prompt") != "both"]
        print(json.dumps({"arm": arm, "queue": len(items), "insight_in_queue": len(insight), "duplicates": dups,
                          "queued_and_completed": sorted(set(ids) & done), "completed": len(done),
                          "completed_insight": sorted(d for d in done if "-insight-" in d),
                          "in_flight_or_missing": in_flight_or_missing, "bad_insight_argv": bad_insight,
                          "non_insight_not_both": bad_other, "failed": failed}))
    calls = json.loads((out / "calls.json").read_text())
    for name, object_ids in calls.items():
        states = []
        for object_id in object_ids:
            try:
                modal.FunctionCall.from_id(object_id).get(timeout=0)
                states.append("finished")
            except TimeoutError:
                states.append("running")
            except Exception as error:  # cancelled workers raise here
                states.append(type(error).__name__)
        print(name, dict(collections.Counter(states)))


if __name__ == "__main__":
    main()
