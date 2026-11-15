"""Write the pre-registered step-2 config and job list for memory v2 (adaptive revisit episodes).

Case rule (fixed before any v2 adaptive episode exists): the GC and out-of-sight cases whose full1
budget-memory episode contained at least one revisit chunk beyond the stock window in the audit
(reproduction/full-run-memory-audit/revisits.json on main; 22 GC + 14 out-of-sight). Flags are the
final1 stock arm's flags for the same case plus the memory-v2 flags; the comparison arm is the
final1 stock episode of the same case, seed and hardware (Lambda H100 PCIe).
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def memory_flags(source):
    return ["--memory-v2", source, "--memv2-select", "verified", "--memv2-retrieved", "10", "--memv2-rope", "native"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revisits", type=Path, required=True, help="revisits.json from the full-run memory audit")
    parser.add_argument("--step1-commit", default="", help="code commit validated on the fixed paths")
    parser.add_argument("--primary", choices=["disk", "recompute"], required=True, help="K/V source of the primary arm")
    parser.add_argument("--reference", type=int, default=8,
                        help="number of held-out cases (first in order) also run with the other K/V source")
    parser.add_argument("--step1-evidence", default="", help="one-line summary of the fixed-path result behind the choice")
    parser.add_argument("--step1-jobs-in", default="", help="comma-separated step-1 job ids that were complete at freeze time")
    parser.add_argument("--frozen-at", default="")
    args = parser.parse_args()
    audit = json.loads(args.revisits.read_text())
    cases = sorted((name.split("-")[1], name.split("-")[2]) for name in
                   (e["episode"] for e in audit["episodes"] if e["revisit_chunks"] > 0)
                   if name.split("-")[1] in ("gc", "outsight"))
    final = json.loads((ROOT / "reproduction/final-run-config.json").read_text())
    development = {(c["split"], c["task_id"]) if isinstance(c, dict) else tuple(c.split(":"))
                   for c in final["development_cases_used_to_choose_the_method"]} \
        if not isinstance(final["development_cases_used_to_choose_the_method"], dict) else set()
    if isinstance(final["development_cases_used_to_choose_the_method"], dict):
        for split, tasks in final["development_cases_used_to_choose_the_method"].items():
            if isinstance(tasks, list):
                development |= {(split, task) for task in tasks}
    development |= {("gc", "GC007"), ("gc", "GC001")}  # step-1 fixed-path scenes
    stock = {(j["split"], j["task_id"]): j for j in map(json.loads, (ROOT / "reproduction/final-run-jobs.jsonl").read_text().splitlines())
             if j["arm"] == "stock"}
    jobs = []
    # Held-out cases first, GC and out-of-sight interleaved; development cases last.
    ordered = sorted(cases, key=lambda c: (c in development, cases.index(c) % 2, cases.index(c)))
    for split, task in ordered:
        base = stock[split, task]
        name = f"memv2-{split}-{task}-seed42"
        argv = [*base["argv"], *memory_flags(args.primary)]
        jobs.append({"id": name, "arm": "memv2", "kv_source": args.primary, "split": split, "task_id": task, "seed": 42,
                     "development": (split, task) in development, "needs_player": True, "argv": argv,
                     "output": f"outputs/memv2-step2/{name}", "stock_episode": base["id"],
                     "command": "python -m worldsync.lingbot_rollout " + " ".join(argv) + f" --output outputs/memv2-step2/{name}"})
    other = "disk" if args.primary == "recompute" else "recompute"
    reference = []
    for split, task in [c for c in ordered if c not in development][:args.reference]:
        base = stock[split, task]
        name = f"memv2{other}-{split}-{task}-seed42"
        argv = [*base["argv"], *memory_flags(other)]
        reference.append({"id": name, "arm": f"memv2-{other}-reference", "kv_source": other, "split": split, "task_id": task,
                          "seed": 42, "development": False, "needs_player": True, "argv": argv,
                          "output": f"outputs/memv2-step2/{name}", "stock_episode": base["id"],
                          "command": "python -m worldsync.lingbot_rollout " + " ".join(argv) + f" --output outputs/memv2-step2/{name}"})
    jobs += reference  # reference jobs run after the primary arm
    (ROOT / "reproduction/memory-v2-step2-jobs.jsonl").write_text("".join(json.dumps(j) + "\n" for j in jobs))
    config = {
        "label": "memv2-step2", "written_before_any_v2_adaptive_episode": True,
        "purpose": "Does memory v2 (unlimited exact-KV archive on disk, content-verified retrieval) improve adaptive "
                   "episodes that actually contain revisits beyond the stock window? Exploratory follow-up to the full-run null.",
        "code_commit_validated_on_fixed_paths": args.step1_commit,
        "case_rule": "GC and out-of-sight cases whose full1 budget-memory episode had >= 1 revisit chunk in "
                     "reproduction/full-run-memory-audit/revisits.json (cost threshold 0.5); fixed before generation",
        "cases": [{"split": s, "task_id": t, "development": (s, t) in development} for s, t in ordered],
        "counts": {"gc": sum(s == "gc" for s, _ in ordered), "outsight": sum(s == "outsight" for s, _ in ordered),
                   "development": sum(c in development for c in ordered)},
        "frozen_at": args.frozen_at,
        "step1_status_at_freeze": {"note": "step 1 was only partly complete when this file was frozen",
                                   "jobs_complete": sorted(filter(None, args.step1_jobs_in.split(","))),
                                   "jobs_total": 27},
        "primary_kv_source": args.primary, "primary_choice_evidence": args.step1_evidence,
        "reference_arm": f"{other} K/V on the first {len(reference)} held-out cases (sanity only: player trajectories differ "
                         "between runs, so recompute-vs-disk parity is measured on the fixed paths of step 1, not here)",
        "arm": {"flags": "final1 stock flags for the case + " + " ".join(memory_flags(args.primary)), "seed": 42,
                "hardware": "Lambda gpu_1x_h100_pcie", "player": "claude-haiku-4-5-20251001 via the local proxy tunnel"},
        "comparison_arm": "final1 stock episode of the same case (same seed, flags, hardware, code lineage abfa73a)",
        "development_data": "the fixed paths of step 1 (GC007, GC001), the 20 final1 development cases, and the full1 "
                            "stock videos used for the offline replay; thresholds (60 deg / 2 units pose match, DINO 0.75 place "
                            "verification, 0.80 direct match) were chosen on those and are frozen here",
        "metrics": {
            "judge_independent": "DINO(start frame, frame at each commanded return to the start pose), paired by case with the "
                                 "stock episode where both arms contain a return; mean and bootstrap 95% CI",
            "judge": "released PlayWorld rubric/gate/aggregation, gemini-3.1-pro-preview on Vertex, 3 independent calls per "
                     "video for BOTH arms; paired mean-of-3 difference with bootstrap 95% CI over cases; all cases and "
                     "held-out (non-development) cases reported separately; per dimension (GC, out-of-sight)",
            "mechanism": "chunks where retrieval replaced attention, frames retrieved, disk bytes, read seconds, added seconds per chunk"},
        "known_limits": ["one seed", "player trajectories differ between arms, so pairs compare episodes, not identical paths",
                         "the case rule was derived from the budget arm's trajectories on Modal",
                         "v2 cannot make the model undo translation; it only stops injecting mismatched views",
                         "a positive result would be exploratory: it is not the pre-registered final1 comparison"]}
    (ROOT / "reproduction/memory-v2-config.json").write_text(json.dumps(config, indent=1) + "\n")
    print(len(jobs), "jobs (", len(jobs) - len(reference), "primary +", len(reference), "reference );", config["counts"])


if __name__ == "__main__":
    main()
