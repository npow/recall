"""Scoring, status and analysis for the frozen final run `final1` (see reproduction/final-run-config.json).

    python scripts/final_run_score.py watch      # loop: judge new episodes (call 1 first, calls 2-3 when idle), write status
    python scripts/final_run_score.py status     # write reproduction/final-run-status.json once
    python scripts/final_run_score.py analyze    # write reproduction/final-run/{analysis.json,table.md,per-episode.csv}

Episodes are read from outputs/final1/<id>/{rollout.mp4,run.json} (synced back from the GPU machines). Judge: released
PlayWorld rubric, gate and aggregation via worldsync.gemini_vertex_judge (Vertex, gcloud login; tokens never printed).
Each repetition has its own directory and cache, so the three calls are independent. The judge sees a neutral model name.
"""
import argparse
import csv
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EPISODES = ROOT / "outputs/final1"
JUDGE = ROOT / "outputs/final1-judge"
SPLITS = ("gc", "if", "insight", "outsight")
NAME = {"gc": "GC", "if": "IF", "insight": "insight", "outsight": "OE"}
PAPER = {"LingBot-World2 (hosted, paper)": {"gc": 2.04, "if": 2.13, "insight": 1.95, "outsight": 1.16, "overall": 1.82},
         "Genie 3 (paper)": {"overall": 2.12}}


def jobs():
    return [json.loads(line) for line in (ROOT / "reproduction/final-run-jobs.jsonl").read_text().splitlines() if line.strip()]


def completed_episodes():
    rows = {}
    for job in jobs():
        directory = EPISODES / job["id"]
        run, video = directory / "run.json", directory / "rollout.mp4"
        if not (run.is_file() and video.is_file()):
            continue
        try:
            if json.loads(run.read_text()).get("status") != "completed":
                continue
        except ValueError:
            continue
        rows[job["id"]] = job
    return rows


def manifest_rows(identifiers, done):
    from worldsync import gemini_vertex_judge
    rows = []
    for identifier in identifiers:
        job = done[identifier]
        record = gemini_vertex_judge.load_dataset_task(ROOT / "external/PlayWorld", job["split"], job["task_id"])
        rows.append({"episode_id": identifier, "split": job["split"], "task_id": job["task_id"],
                     "video": str((EPISODES / identifier / "rollout.mp4").resolve()),
                     "reference_image": str((ROOT / "external/PlayWorld/data" / record["image_path"]).resolve()),
                     "world_model": "LingBot-World2 14B causal-fast"})
    return rows


def result(rep, identifier):
    path = JUDGE / f"rep{rep}" / "episodes" / f"{identifier}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


def score_of(rep, identifier):
    data = result(rep, identifier)
    if not data or data.get("error"):
        return None
    value = (data.get("scores") or {}).get("official")
    return None if value is None else float(value)


def judge(rep, identifiers, done, workers):
    from worldsync import gemini_vertex_judge
    directory = JUDGE / f"rep{rep}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / f"manifest-{int(time.time())}.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in manifest_rows(identifiers, done)))
    gemini_vertex_judge.run_manifest(manifest, directory, playworld=ROOT / "external/PlayWorld", workers=workers)


def status():
    done = completed_episodes()
    all_jobs = jobs()
    out = {"label": "final1", "updated": time.strftime("%Y-%m-%d %H:%M:%S PST"), "expected": len(all_jobs), "arms": {}}
    for arm in ("stock", "director"):
        entry = {"expected": {}, "generated": {}, "scored_call1": {}, "scored_3_calls": {}}
        for split in SPLITS:
            ids = [job["id"] for job in all_jobs if job["arm"] == arm and job["split"] == split]
            if not ids:
                continue
            entry["expected"][split] = len(ids)
            entry["generated"][split] = sum(i in done for i in ids)
            entry["scored_call1"][split] = sum(score_of(1, i) is not None for i in ids)
            entry["scored_3_calls"][split] = sum(all(score_of(rep, i) is not None for rep in (1, 2, 3)) for i in ids)
        for key in ("expected", "generated", "scored_call1", "scored_3_calls"):
            entry[key + "_total"] = sum(entry[key].values())
        out["arms"][arm] = entry
    errors = []
    for identifier in done:
        data = result(1, identifier)
        if data and data.get("error"):
            errors.append({"episode": identifier, "error": str(data["error"])[:200]})
    out["judge_errors_call1"] = errors
    (ROOT / "reproduction/final-run-status.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def watch(args):
    while not (ROOT / "outputs/final1-watch.stop").exists():
        done = completed_episodes()
        todo1 = [i for i in done if score_of(1, i) is None]
        try:
            if todo1:
                judge(1, todo1, done, args.workers)
            else:  # idle on call 1: spend the time on the secondary calls
                for rep in (2, 3):
                    extra = [i for i in done if score_of(rep, i) is None][:48]
                    if extra:
                        judge(rep, extra, done, args.workers)
                        break
        except RuntimeError as error:  # expired gcloud session and similar: wait, never crash
            print(json.dumps({"event": "judge_error", "error": str(error)[:200]}), flush=True)
            time.sleep(120)
        report = status()
        print(json.dumps({"event": "cycle", "time": report["updated"],
                          **{arm: [v["generated_total"], v["scored_call1_total"], v["scored_3_calls_total"]]
                             for arm, v in report["arms"].items()}}), flush=True)
        time.sleep(args.interval)


def bootstrap(values_by_split, draws=4000, seed=11):
    """CI for the equal mean of the dimension means, resampling cases within each dimension."""
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        means.append(statistics.mean(statistics.mean(rng.choices(values, k=len(values))) for values in values_by_split.values()))
    means.sort()
    return [round(means[int(0.025 * draws)], 3), round(means[int(0.975 * draws) - 1], 3)]


def arm_scores(reps):
    """{arm: {split: {task: score}}}; the director arm's GC entries are the stock arm's (defined identical)."""
    table = {"stock": {s: {} for s in SPLITS}, "director": {s: {} for s in SPLITS}}
    for job in jobs():
        values = [score_of(rep, job["id"]) for rep in reps]
        if any(value is None for value in values):
            continue
        table[job["arm"]][job["split"]][job["task_id"]] = statistics.mean(values)
    table["director"]["gc"] = dict(table["stock"]["gc"])
    return table


def summarize(table, keep):
    out = {}
    paired_cases = {split: sorted(t for t in table["stock"][split] if t in table["director"][split] and keep(split, t)) for split in SPLITS}
    for arm in ("stock", "director"):
        by_split = {split: [table[arm][split][t] for t in paired_cases[split]] for split in SPLITS}
        if not all(by_split.values()):
            out[arm] = None
            continue
        dims = {split: round(statistics.mean(values), 3) for split, values in by_split.items()}
        out[arm] = {"n": {split: len(values) for split, values in by_split.items()}, "dimension_means": dims,
                    "overall": round(statistics.mean(dims.values()), 3), "overall_ci95": bootstrap(by_split),
                    "floor_count": {split: sum(v <= 1.0 + 1e-9 for v in values) for split, values in by_split.items()}}
    if out["stock"] and out["director"]:
        deltas = {split: [table["director"][split][t] - table["stock"][split][t] for t in paired_cases[split]] for split in SPLITS}
        out["paired_director_minus_stock"] = {
            "dimension_delta": {split: round(statistics.mean(values), 3) for split, values in deltas.items()},
            "overall_delta": round(statistics.mean(statistics.mean(values) for values in deltas.values()), 3),
            "overall_delta_ci95": bootstrap(deltas),
            "wins_ties_losses": {split: [sum(v > 1e-9 for v in values), sum(abs(v) <= 1e-9 for v in values), sum(v < -1e-9 for v in values)]
                                 for split, values in deltas.items()},
            "note": "GC delta is zero by construction (the director arm's GC episodes are the stock arm's)"}
    return out


def analyze(args):
    config = json.loads((ROOT / "reproduction/final-run-config.json").read_text())
    development = {(split, task) for split, tasks in config["development_cases_used_to_choose_the_method"].items()
                   if isinstance(tasks, list) for task in tasks}
    sets = {"all_171": lambda s, t: True, "held_out_151": lambda s, t: (s, t) not in development,
            "all_without_GC013": lambda s, t: t != "GC013",
            "held_out_without_GC013": lambda s, t: (s, t) not in development and t != "GC013"}
    report = {"label": "final1", "generated": time.strftime("%Y-%m-%d %H:%M:%S PST"), "paper": PAPER, "status": status()}
    for name, reps in (("primary_single_call", (1,)), ("secondary_mean_of_3", (1, 2, 3))):
        table = arm_scores(reps)
        report[name] = {key: summarize(table, keep) for key, keep in sets.items()}
    out = ROOT / "reproduction/final-run"
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(report, indent=1) + "\n")
    with (out / "per-episode.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "arm", "split", "task_id", "development", "call1", "call2", "call3", "gate_call1"])
        for job in jobs():
            first = result(1, job["id"]) or {}
            gate = (first.get("gate") or {})
            writer.writerow([job["id"], job["arm"], job["split"], job["task_id"], job["development"],
                             *[score_of(rep, job["id"]) for rep in (1, 2, 3)],
                             gate.get("verdict", gate.get("instruction_following", ""))])
    lines = ["# final1: stock vs static text director (frozen config: reproduction/final-run-config.json)", "",
             "Overall = equal mean of the four dimension means; 95% CI by bootstrap over cases within each dimension.",
             "Local checkpoint, Lambda H100, Haiku player, Gemini `gemini-3.1-pro-preview`; see the config for deviations.", ""]
    for name in ("primary_single_call", "secondary_mean_of_3"):
        lines += [f"## {name.replace('_', ' ')}", "", "| set | arm | GC | IF | insight | OE | overall [95% CI] | n (GC/IF/insight/OE) |",
                  "|---|---|---:|---:|---:|---:|---|---|"]
        for key, summary in report[name].items():
            for arm in ("stock", "director"):
                row = summary.get(arm)
                if not row:
                    lines.append(f"| {key} | {arm} | incomplete | | | | | |")
                    continue
                d, n = row["dimension_means"], row["n"]
                lines.append(f"| {key} | {arm} | {d['gc']:.2f} | {d['if']:.2f} | {d['insight']:.2f} | {d['outsight']:.2f} | "
                             f"{row['overall']:.2f} [{row['overall_ci95'][0]:.2f}, {row['overall_ci95'][1]:.2f}] | "
                             f"{n['gc']}/{n['if']}/{n['insight']}/{n['outsight']} |")
            paired = summary.get("paired_director_minus_stock")
            if paired:
                d = paired["dimension_delta"]
                lines.append(f"| {key} | director - stock | {d['gc']:+.2f} | {d['if']:+.2f} | {d['insight']:+.2f} | {d['outsight']:+.2f} | "
                             f"{paired['overall_delta']:+.2f} [{paired['overall_delta_ci95'][0]:+.2f}, {paired['overall_delta_ci95'][1]:+.2f}] | paired |")
        lines.append("")
    lines += ["Published references: LingBot-World2 hosted 1.82 (GC 2.04, IF 2.13, insight 1.95, OE 1.16); Genie 3 2.12.", ""]
    (out / "table.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["watch", "status", "analyze"])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--interval", type=int, default=45)
    args = parser.parse_args()
    if args.command == "watch":
        watch(args)
    elif args.command == "status":
        print(json.dumps(status(), indent=1))
    else:
        analyze(args)


if __name__ == "__main__":
    main()
