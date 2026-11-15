"""Revisit-horizon benchmark: how long can the generated world be left unseen and still be found again?

Scripted camera path per scene: hold the start view (1 chunk), turn right about 180 degrees (3 chunks at
68 deg/s commanded), stay away for H seconds, turn back left (3 chunks), hold (1 chunk). Arms: stock
LingBot-World2 14B causal-fast and exact bounded pose memory. Fixed actions, no player.

Commands:
  freeze   write reproduction/revisit-horizon-config.json and reproduction/revisit-horizon-jobs.jsonl
  score    judge landed videos (released PlayWorld Gemini verifier, synthetic objective) and compute DINO
  analyze  paired tables, bootstrap CIs, plot, example strips

Not a PlayWorld leaderboard score: the objective and actions are scripted, the rubric questions are each
GC task's released questions applied to our scripted objective.
"""
import argparse
import concurrent.futures
import hashlib
import json
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAYWORLD = ROOT / "external/PlayWorld"
CONFIG = ROOT / "reproduction/revisit-horizon-config.json"
JOBS = ROOT / "reproduction/revisit-horizon-jobs.jsonl"
RESULTS = ROOT / "outputs/revisit-horizon"
JUDGE = ROOT / "outputs/revisit-horizon-judge"
REPORT = ROOT / "reproduction/revisit-horizon"

SALT = "revisit-horizon-2026-11-15"
SEED = 42
SCENES = 20
HORIZONS = [12, 24, 6, 0]  # stage order: most informative first
TURN_CHUNKS = 3
TURN_RATE = 68
JUDGE_CALLS = 2
# Scenes used while developing or validating any memory method on scripted paths, plus the GC
# development cases of the final1 run. They are excluded from selection.
DEVELOPMENT_SCENES = ["GC001", "GC006", "GC007", "GC010", "GC013", "GC027", "GC033", "GC040", "GC045", "GC046"]
ARMS = {
    "stock": [],
    "memory": ["--budget-memory", "--budget-rope", "native", "--budget-retrieved", "10",
               "--budget-store", "exact", "--budget-layers", "all"],
}


def gc_tasks():
    return {record["task_id"]: record for record in json.loads((PLAYWORLD / "data/gc/data.json").read_text())}


def select_scenes():
    pool = [task for task in gc_tasks() if task not in DEVELOPMENT_SCENES]
    order = sorted(pool, key=lambda task: hashlib.sha256(f"{SALT}:{task}".encode()).hexdigest())
    return order[:SCENES], order[SCENES:]


def actions(horizon):
    return ["WAIT"] + ["RIGHT"] * TURN_CHUNKS + ["WAIT"] * horizon + ["LEFT"] * TURN_CHUNKS + ["WAIT"]


def objective(horizon):
    wait = f"wait for {horizon} seconds, " if horizon else ""
    return f"Turn right 180 degrees, {wait}then turn back to the original view."


def action_steps(horizon):
    steps = ["hold(Right,3000ms)"] + ([f"wait({horizon * 1000}ms)"] if horizon else []) + ["hold(Left,3000ms)"]
    return steps


def job(arm, task, horizon, stage):
    plan = actions(horizon)
    identifier = f"rh-{arm}-{task}-H{horizon:02d}-seed{SEED}"
    argv = ["--split", "gc", "--task-id", task, "--seed", str(SEED), "--max-chunks", str(len(plan)),
            "--native-fixed-controls", "--fixed-actions", *plan, "--turn-degrees-per-second", str(TURN_RATE),
            "--offload-text-encoder", *ARMS[arm]]
    output = f"outputs/revisit-horizon/{identifier}"
    return {"id": identifier, "arm": arm, "split": "gc", "task_id": task, "horizon_s": horizon, "seed": SEED,
            "stage": stage, "needs_player": False, "argv": argv, "output": output, "chunks": len(plan),
            "command": "python -m worldsync.lingbot_rollout " + shlex.join(argv) + f" --output {output}"}


def build_jobs():
    scenes, _ = select_scenes()
    rows = []
    for stage, horizon in enumerate(HORIZONS):
        for task in scenes:  # hash order: a truncated run still covers a hash-order prefix of scenes
            for arm in ARMS:
                rows.append(job(arm, task, horizon, stage))
    return rows


def freeze(_):
    scenes, reserve = select_scenes()
    rows = build_jobs()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config = {
        "name": "revisit-horizon",
        "frozen_before_generation": True,
        "question": "How long can the camera look away before the generated world is no longer found again on return, "
                    "and does archived-view memory extend that horizon?",
        "model": "LingBot-World2 14B causal-fast (stock weights, no training)",
        "code_base_commit": commit,
        "hardware": "Lambda gpu_1x_h100_pcie (both arms on the same GPU model)",
        "seed": SEED,
        "scene_selection": {
            "rule": f"GC split, sort by sha256('{SALT}:' + task_id), first {SCENES} after excluding development scenes",
            "excluded_development_scenes": DEVELOPMENT_SCENES,
            "scenes": scenes, "next_in_hash_order_not_run": reserve[:4],
        },
        "path": {
            "actions": "WAIT, RIGHT x3, WAIT x H, LEFT x3, WAIT (one action = one 1-second chunk; native fixed controls)",
            "turn_degrees_per_second": TURN_RATE,
            "turn_note": "3 chunks at 68 deg/s commanded; realized rotation was measured at about 0.9 of commanded on "
                         "GC007/GC001, i.e. about 180 degrees. Decided on those two development scenes only.",
            "horizons_seconds_away": sorted(HORIZONS),
            "chunks_by_horizon": {str(h): len(actions(h)) for h in sorted(HORIZONS)},
            "text_prompt": "caption (rollout default)",
        },
        "arms": {
            "stock": {"flags": [], "note": "rolling window 18 latent frames incl. 6 sink frames (about 4.5 s)"},
            "memory": {"flags": ARMS["memory"],
                       "note": "exact bounded pose memory: 24-frame archive, pose-redundancy eviction, 10 retrieved frames, "
                               "native RoPE, same attended budget. Valid at every horizon here because the frames added "
                               "during the stationary hold are pose duplicates and are evicted first; the outbound sweep "
                               "and start frames stay archived. Chosen on GC007/GC001 before this run."},
            "memory_v2_disk": "not included: it had not passed its fixed-path step 1 when this file was frozen",
        },
        "judge": {
            "primary": "released PlayWorld verifier (gate + rubric + released aggregation) through worldsync.gemini_vertex_judge, "
                       "Vertex gemini-3.1-pro-preview",
            "task_record": "each scene's released GC record (image, caption, policies, its own rubric questions) with `prompt`, "
                           "`action_sequence`, `action_sequence_steps` and `action` replaced by the scripted objective below",
            "objective_template": {str(h): objective(h) for h in sorted(HORIZONS)},
            "action_sequence_template": {str(h): "; ".join(action_steps(h)) for h in sorted(HORIZONS)},
            "calls_per_video": JUDGE_CALLS, "score_per_video": "mean of the calls' official scores",
            "world_model_name_shown_to_judge": "LingBot-World-Infinity (same for both arms; no arm label)",
            "reference_image": "the task's released first image",
        },
        "metrics": {
            "primary": "judged official score (1-5), mean of 2 calls",
            "secondary": ["DINOv2-small CLS cosine, final frame vs frame 12 (end of the opening hold) and vs frame 0",
                          "judge field revisit_observed (yes counts per arm)", "gate verdict counts"],
        },
        "analysis": {
            "paired": "memory minus stock per horizon over scenes with both arms complete; bootstrap 95% CI over scenes "
                      "(10000 draws, seed 11); wins/ties/losses",
            "curve": "mean score and mean DINO vs horizon for both arms with bootstrap CIs (PNG)",
            "pooled": "paired difference pooled over horizons 6, 12, 24 with scenes as the resampling unit",
            "truncation_rule": "generation order is horizon 12, 24, 6, 0, scenes in hash order, stock then memory. If the run is "
                               "cut at the 12:00 PST Lambda hard stop, each horizon is reported on the scenes whose two arms "
                               "both completed, with the count stated; no horizon is dropped silently.",
            "examples": "strips for 4 scenes chosen by rule at horizon 12: the two largest and the two smallest paired "
                        "judged differences (ties broken by hash order)",
        },
        "prediction_on_record": "On development scenes memory lifted DINO at H=6 from about 0.3-0.4 to about 0.78. We expect a "
                                "positive paired DINO difference at H>=6 and little difference at H=0 (the start view is still "
                                "inside the stock window). Whether the judged score moves is unknown.",
        "limits": ["scripted actions and objective, not the PlayWorld agent player: not a leaderboard score",
                   "rotation-only path; the audit found forward-then-back returns are not helped",
                   "one seed", "Gemini preview model on Vertex; within-video judge SD was about 0.35 in earlier runs"],
        "jobs": {"file": "reproduction/revisit-horizon-jobs.jsonl", "count": len(rows),
                 "chunks": sum(row["chunks"] for row in rows)},
    }
    CONFIG.write_text(json.dumps(config, indent=2) + "\n")
    JOBS.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(len(rows), "jobs", config["jobs"]["chunks"], "chunks; scenes:", " ".join(scenes))


def synthetic_task(task, horizon):
    record = dict(gc_tasks()[task])
    record.update(prompt=objective(horizon), action_sequence="; ".join(action_steps(horizon)),
                  action_sequence_steps=action_steps(horizon),
                  action="→*8" + (f" -> wait*{horizon}s" if horizon else "") + " -> ←*8",
                  action_sequence_derived_from_action=False)
    return record


def completed():
    rows = [json.loads(line) for line in JOBS.read_text().splitlines() if line.strip()]
    done = []
    for row in rows:
        directory = RESULTS / row["id"]
        run, video = directory / "run.json", directory / "rollout.mp4"
        if run.is_file() and video.is_file() and video.stat().st_size > 0:
            try:
                if json.loads(run.read_text()).get("status") == "completed":
                    done.append(row)
            except json.JSONDecodeError:
                pass
    return rows, done


def judge_one(client, row, rep):
    from worldsync.gemini_vertex_judge import judge_episode
    target = JUDGE / f"rep{rep}" / "episodes" / f"{row['id']}.json"
    if target.is_file():
        previous = json.loads(target.read_text())
        if not previous.get("error") and (previous.get("scores") or {}).get("official") is not None:
            return previous
    task = synthetic_task(row["task_id"], row["horizon_s"])
    video = RESULTS / row["id"] / "rollout.mp4"
    try:
        result = judge_episode(client, PLAYWORLD, task, video, PLAYWORLD / "data" / task["image_path"],
                               cache_dir=JUDGE / f"rep{rep}" / "cache")
    except Exception as error:  # keep failures in the matrix
        result = {"task_id": row["task_id"], "error": f"{type(error).__name__}: {str(error)[:300]}"}
    result.update(episode_id=row["id"], arm=row["arm"], horizon_s=row["horizon_s"], objective=objective(row["horizon_s"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=1) + "\n")
    return result


_dino = {}


def dino(row):
    """DINOv2-small CLS cosine of the final frame against frame 12 and frame 0."""
    target = JUDGE / "dino" / f"{row['id']}.json"
    if target.is_file():
        return json.loads(target.read_text())
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    if not _dino:
        _dino["processor"] = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        _dino["model"] = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    directory = RESULTS / row["id"]
    frames = json.loads((directory / "run.json").read_text())["generated_frames"]
    work = JUDGE / "dino" / "frames" / row["id"]
    work.mkdir(parents=True, exist_ok=True)

    def embed(index):
        path = work / f"f{index}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(directory / "rollout.mp4"), "-vf",
                        f"select=eq(n\\,{index})", "-frames:v", "1", str(path)], check=True)
        with torch.no_grad():
            out = _dino["model"](**_dino["processor"](images=Image.open(path).convert("RGB"), return_tensors="pt"))
        return torch.nn.functional.normalize(out.last_hidden_state[:, 0], dim=-1)[0]

    end, start, first = embed(frames - 1), embed(12), embed(0)
    away = embed(16 * (1 + TURN_CHUNKS) + 4)  # just after the outbound turn
    result = {"id": row["id"], "generated_frames": frames, "end_vs_frame12": float(end @ start),
              "end_vs_frame0": float(end @ first), "away_vs_frame12": float(away @ start)}
    target.write_text(json.dumps(result) + "\n")
    return result


def score(args):
    from worldsync.gemini_vertex_judge import create_vertex_client
    while True:
        rows, done = completed()
        client = create_vertex_client()  # fresh credentials every cycle
        pending = [(row, rep) for rep in range(1, JUDGE_CALLS + 1) for row in done
                   if not (JUDGE / f"rep{rep}" / "episodes" / f"{row['id']}.json").is_file()
                   or json.loads((JUDGE / f"rep{rep}" / "episodes" / f"{row['id']}.json").read_text()).get("error")]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(lambda item: judge_one(client, *item), pending):
                print(json.dumps({"episode": result["episode_id"], "gate": (result.get("gate") or {}).get("verdict"),
                                  "score": (result.get("scores") or {}).get("official"), "error": result.get("error")}),
                      flush=True)
        for row in done:
            try:
                dino(row)
            except Exception as error:
                print(json.dumps({"dino_error": row["id"], "error": str(error)[:200]}), flush=True)
        judged = sum((JUDGE / f"rep{rep}" / "episodes" / f"{row['id']}.json").is_file()
                     for rep in range(1, JUDGE_CALLS + 1) for row in done)
        print(json.dumps({"event": "cycle", "time": time.strftime("%H:%M:%S"), "jobs": len(rows),
                          "generated": len(done), "judgements": judged, "of": len(done) * JUDGE_CALLS}), flush=True)
        if not args.watch or (len(done) == len(rows) and judged == len(rows) * JUDGE_CALLS):
            return
        time.sleep(args.interval)


def load_scores():
    rows, done = completed()
    table = {}
    for row in done:
        calls = []
        for rep in range(1, JUDGE_CALLS + 1):
            path = JUDGE / f"rep{rep}" / "episodes" / f"{row['id']}.json"
            if path.is_file():
                data = json.loads(path.read_text())
                if not data.get("error") and (data.get("scores") or {}).get("official") is not None:
                    calls.append(data)
        dino_path = JUDGE / "dino" / f"{row['id']}.json"
        if len(calls) < JUDGE_CALLS or not dino_path.is_file():
            continue
        gates = [(call.get("gate") or {}) for call in calls]
        table[(row["task_id"], row["horizon_s"], row["arm"])] = {
            "id": row["id"], "score": sum(call["scores"]["official"] for call in calls) / len(calls),
            "call_scores": [call["scores"]["official"] for call in calls],
            "gate": [gate.get("verdict") for gate in gates],
            "revisit_observed": [gate.get("revisit_observed") for gate in gates],
            **{key: value for key, value in json.loads(dino_path.read_text()).items() if key != "id"}}
    return rows, done, table


def bootstrap(values, draws=10000, seed=11):
    if not values:
        return None
    rng = random.Random(seed)
    count = len(values)
    means = sorted(sum(values[rng.randrange(count)] for _ in range(count)) / count for _ in range(draws))
    return [means[int(0.025 * draws)], means[int(0.975 * draws) - 1]]


def summarize(values):
    return {"n": len(values), "mean": sum(values) / len(values) if values else None, "ci95": bootstrap(values)}


def analyze(args):
    rows, done, table = load_scores()
    scenes, _ = select_scenes()
    report = {"jobs": len(rows), "generated": len(done), "fully_scored": len(table), "horizons": {}}
    pooled = {"score": {}, "dino": {}}
    for horizon in sorted(HORIZONS):
        paired = [task for task in scenes if all((task, horizon, arm) in table for arm in ARMS)]
        entry = {"paired_scenes": len(paired), "scenes": paired}
        for arm in ARMS:
            cells = [table[(task, horizon, arm)] for task in paired]
            entry[arm] = {
                "score": summarize([cell["score"] for cell in cells]),
                "dino_end_vs_frame12": summarize([cell["end_vs_frame12"] for cell in cells]),
                "dino_end_vs_frame0": summarize([cell["end_vs_frame0"] for cell in cells]),
                "dino_away_vs_frame12": summarize([cell["away_vs_frame12"] for cell in cells]),
                "gate": {verdict: sum(cell["gate"].count(verdict) for cell in cells) for verdict in ("pass", "partial", "fail")},
                "revisit_observed_yes": sum(cell["revisit_observed"].count("yes") for cell in cells),
                "judge_calls": len(cells) * JUDGE_CALLS,
                "at_floor_1.0": sum(cell["score"] == 1.0 for cell in cells)}
        for metric, key in (("score", "score"), ("dino", "end_vs_frame12")):
            deltas = [table[(task, horizon, "memory")][key] - table[(task, horizon, "stock")][key] for task in paired]
            margin = 1e-9 if metric == "score" else 0.02
            entry[f"paired_{metric}"] = {**summarize(deltas), "wins": sum(d > margin for d in deltas),
                                         "ties": sum(abs(d) <= margin for d in deltas), "losses": sum(d < -margin for d in deltas)}
            if horizon >= 6:
                for task, delta in zip(paired, deltas):
                    pooled[metric].setdefault(task, []).append(delta)
        report["horizons"][str(horizon)] = entry
    report["pooled_h6_h12_h24"] = {metric: summarize([sum(values) / len(values) for values in by_scene.values()])
                                   for metric, by_scene in pooled.items()}
    report["per_episode"] = [{"task_id": task, "horizon_s": horizon, "arm": arm, **cell}
                             for (task, horizon, arm), cell in sorted(table.items())]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "analysis.json").write_text(json.dumps(report, indent=1) + "\n")
    lines = ["| away (s) | scenes | stock score | memory score | paired score Δ [95% CI] (W/T/L) | stock DINO | memory DINO | paired DINO Δ [95% CI] (W/T/L) | revisit seen, stock / memory |",
             "|---:|---:|---:|---:|---|---:|---:|---|---|"]

    def interval(item, digits=2):
        if item["mean"] is None:
            return "n/a"
        return f"{item['mean']:+.{digits}f} [{item['ci95'][0]:+.{digits}f}, {item['ci95'][1]:+.{digits}f}]"

    for horizon in sorted(HORIZONS):
        entry = report["horizons"][str(horizon)]
        if not entry["paired_scenes"]:
            lines.append(f"| {horizon} | 0 | | | | | | | |")
            continue
        s, m = entry["stock"], entry["memory"]
        ps, pd = entry["paired_score"], entry["paired_dino"]
        lines.append(f"| {horizon} | {entry['paired_scenes']} | {s['score']['mean']:.2f} | {m['score']['mean']:.2f} | "
                     f"{interval(ps)} ({ps['wins']}/{ps['ties']}/{ps['losses']}) | {s['dino_end_vs_frame12']['mean']:.2f} | "
                     f"{m['dino_end_vs_frame12']['mean']:.2f} | {interval(pd)} ({pd['wins']}/{pd['ties']}/{pd['losses']}) | "
                     f"{s['revisit_observed_yes']}/{s['judge_calls']} vs {m['revisit_observed_yes']}/{m['judge_calls']} |")
    pooled_line = "Pooled over 6, 12 and 24 s (scenes as units): score " + interval(report["pooled_h6_h12_h24"]["score"]) + \
                  f" (n={report['pooled_h6_h12_h24']['score']['n']}), DINO " + interval(report["pooled_h6_h12_h24"]["dino"]) + "."
    (REPORT / "table.md").write_text("\n".join(lines) + "\n\n" + pooled_line + "\n")
    print("\n".join(lines))
    print(pooled_line)
    plot(report)
    if args.strips:
        strips(report, table)


def plot(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    horizons = [h for h in sorted(HORIZONS) if report["horizons"][str(h)]["paired_scenes"]]
    if not horizons:
        return
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, key, label in ((axes[0], "score", "Judged score (1-5, mean of 2 Gemini calls)"),
                             (axes[1], "dino_end_vs_frame12", "DINO similarity, final frame vs start view")):
        for arm, colour in (("stock", "tab:gray"), ("memory", "tab:blue")):
            items = [report["horizons"][str(h)][arm][key] for h in horizons]
            means = [item["mean"] for item in items]
            axis.errorbar(horizons, means, yerr=[[m - item["ci95"][0] for m, item in zip(means, items)],
                                                 [item["ci95"][1] - m for m, item in zip(means, items)]],
                          marker="o", capsize=3, color=colour, label=arm)
        axis.set_xlabel("Seconds the start view is out of sight")
        axis.set_ylabel(label)
        axis.set_xticks(horizons)
        axis.grid(alpha=0.3)
        axis.legend()
    counts = ", ".join(f"H={h}: {report['horizons'][str(h)]['paired_scenes']}" for h in horizons)
    figure.suptitle(f"Revisit horizon: stock vs archived-view memory (scenes per horizon: {counts}; bars: 95% bootstrap CI)",
                    fontsize=9)
    figure.tight_layout()
    figure.savefig(REPORT / "horizon.png", dpi=130)


def strips(report, table):
    from PIL import Image, ImageDraw
    entry = report["horizons"]["12"]
    deltas = sorted(((table[(task, 12, "memory")]["score"] - table[(task, 12, "stock")]["score"], index, task)
                     for index, task in enumerate(entry["scenes"])), key=lambda item: (-item[0], item[1]))
    chosen = deltas[:2] + deltas[-2:] if len(deltas) >= 4 else deltas
    width, height, picks = 256, 144, []
    for delta, _, task in chosen:
        sheet = Image.new("RGB", (width * 5, (height + 16) * 2 + 16), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((4, 2), f"{task}, 12 s away: columns = start view (frame 12), end of outbound turn, mid hold, "
                          f"turning back, final frame. Paired judged difference {delta:+.2f}", fill="black")
        for line, arm in enumerate(("stock", "memory")):
            cell = table[(task, 12, arm)]
            directory = RESULTS / cell["id"]
            total = cell["generated_frames"]
            indices = [12, 16 * 4 + 4, total // 2, total - 16 * 2 - 8, total - 1]
            y = 16 + line * (height + 16)
            draw.text((4, y), f"{arm}: score {cell['score']:.2f} (calls {cell['call_scores']}), DINO final vs start "
                              f"{cell['end_vs_frame12']:.2f}, revisit seen {cell['revisit_observed']}", fill="black")
            for column, index in enumerate(indices):
                path = REPORT / "frames" / f"{cell['id']}-{index}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(directory / "rollout.mp4"), "-vf",
                                f"select=eq(n\\,{index})", "-frames:v", "1", str(path)], check=True)
                sheet.paste(Image.open(path).convert("RGB").resize((width, height)), (column * width, y + 16))
        target = REPORT / f"strip-{task}-H12.jpg"
        sheet.save(target, quality=85)
        picks.append({"task_id": task, "paired_score_delta": delta, "file": target.name})
    (REPORT / "strips.json").write_text(json.dumps(picks, indent=1) + "\n")
    import shutil
    shutil.rmtree(REPORT / "frames", ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "score", "analyze"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--strips", action="store_true")
    args = parser.parse_args()
    {"freeze": freeze, "score": score, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()
