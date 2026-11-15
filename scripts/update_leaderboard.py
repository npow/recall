"""Regenerate the repository leaderboard from complete, separate cohorts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    path = ROOT / name
    return json.loads(path.read_text()) if path.exists() else None


def powered_rows():
    """Seed-42 paired Gemini comparison on 14 GC/OE development cases (stock vs budget memory vs 46/6)."""
    data = read("reproduction/powered-comparison/seed42/powered-analysis.json")
    if not data or data.get("judge_errors") or data["episodes_scored"] != 42:
        return None
    names = {"stock": "stock 18/6", "budget": "budget memory (K10 native)", "cache46": "46/6 cache"}
    rows = []
    for arm, label in names.items():
        stats = data["arms"][arm]
        paired = data["paired"].get(arm, {}).get("gemini_all")
        delta = (f"{paired['mean_delta']:+.2f} [{paired['ci95'][0]:+.2f}, {paired['ci95'][1]:+.2f}] "
                 f"({paired['wins']}/{paired['ties']}/{paired['losses']})") if paired else "baseline"
        rows.append(f"| {label} |{stats['n']}|{stats['gemini_mean']:.2f} "
                    f"[{stats['gemini_ci95'][0]:.2f}, {stats['gemini_ci95'][1]:.2f}]|{stats['gc']:.2f}|{stats['oe']:.2f}"
                    f"|{stats['gate_non_fail']}/{stats['n']}|{delta}|")
    return rows


def rescore_rows():
    """Earlier 75-episode Gemini rescoring of the first cohorts; read from the committed arm table."""
    path = ROOT / "reproduction/gemini-rescore/arm-table.md"
    if not path.exists():
        return None
    rows = []
    for line in path.read_text().splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 7 and cells[0] in ("stock", "cache46", "fb", "fbcomp"):
            label = {"stock": "stock 18/6", "cache46": "46/6 cache", "fb": "measured-motion feedback",
                     "fbcomp": "feedback + turn compensation"}[cells[0]]
            rows.append(f"| {label} |{cells[1]}|{cells[2]}|{cells[3]}|{cells[4]}|{cells[5]}|")
    return rows or None


def extra_sections():
    lines = []
    powered = powered_rows()
    if powered:
        lines.extend(["", "## Gemini — paired GC/OE development cohort (seed 42)", "",
            "14 hash-selected GC and out-of-sight cases, three arms, same cases and seed, `--end-guard` and player retries on every arm. "
            "Scores are the Gemini mean over GC and out-of-sight episodes only, **not** the four-dimension overall, "
            "so they are **not comparable with 2.12 or 1.82**. Paired delta is against stock over (case, seed) pairs; "
            "wins/ties/losses in brackets. Every interval includes zero: no arm is established better than stock. "
            "Opus scores the same episodes without seeing the budget-memory effect (+0.02), so judge-specific noise is not ruled out.", "",
            "| Arm | n | Gemini mean [95% CI] | GC | Out-of-sight | Gate non-fail | Paired Δ vs stock [95% CI] (W/T/L) |",
            "|---|---:|---|---:|---:|---:|---|", *powered, "",
            "Evidence: [analysis](reproduction/powered-comparison/seed42/powered-analysis.json), "
            "[table](reproduction/powered-comparison/seed42/powered-table.md), [protocol](reproduction/powered-comparison.md). "
            "A second seed was generated and is being scored; it is added here only when complete."])
    rescore = rescore_rows()
    if rescore:
        lines.extend(["", "## Gemini — earlier rescoring of the first cohorts", "",
            "75 episodes from the first stock / 46/6 / feedback cohorts, rescored with the same Gemini judge (GC and out-of-sight only; "
            "10 generation failures excluded; different player rollouts per arm). A paired 46/6 change of +0.69 [+0.19, +1.23] over 24 pairs "
            "was found here, but the powered cohort above found +0.05 on different episodes, so the effect is **not established**.", "",
            "| Arm | n | GC (n) | Out-of-sight (n) | GC+OE mean | Gate non-fail |", "|---|---:|---:|---:|---:|---:|", *rescore, "",
            "Evidence: [report](reproduction/gemini-rescore.md), [per-episode scores](reproduction/gemini-rescore/per-episode.csv)."])
    lines.extend(["", "## Fixed-path revisit diagnostics (DINO, not a benchmark score)", "",
        "Scripted camera paths with a guaranteed return to the start view; metric is DINO similarity of the final frame to the start view. "
        "Single seeds, crude proxy, two scenes. They show *where* memory helps, not a PlayWorld gain. "
        "On the long 24-chunk absence (GC007): stock about 0.30-0.40, bounded pose memory about 0.77, 46/6 cache about 0.89 (115 GB). "
        "Negative results kept: memory LoRA (degrades ordinary generation), re-anchoring, DeCoPrune ports, prompt/translation tricks, "
        "measured-motion feedback (within noise). See [integration summary](reproduction/INTEGRATION.md), "
        "[revisit matrix](reproduction/revisit/README.md) and [budget memory](reproduction/budgetmem/README.md).",
        "", "## What would change the headline", "",
        "Nothing here is a full-benchmark score. The pass criterion for beating 2.12 was fixed in advance in "
        "[pass-criterion.md](reproduction/pass-criterion.md): equal mean of four dimension means, released Gemini scorer, "
        "identical-harness stock baseline, full 171 cases, lower CI bound above 2.12."])
    return lines


def generate():
    cohorts = []
    definitions = (
        ("Scripted task pilot", "task-pilot-", ("stock", "four-view"), 3),
        ("Adaptive GC027", "adaptive-", ("stock", "memory"), 2),
        ("Start-anchor replay", "start-anchor-", ("stock", "four-view", "start-anchor"), 2),
    )
    for title, prefix, arms, expected in definitions:
        for arm in arms:
            results = [read(str(path.relative_to(ROOT))) for path in
                       (ROOT / "reproduction/gemini-current/episodes").glob(f"{prefix}*-{arm}.json")]
            scored = [row for row in results if not row.get("error")
                      and (row.get("scores") or {}).get("official") is not None]
            complete = len(scored) == expected
            cohorts.append({"cohort": title, "method": arm, "expected": expected, "scored": len(scored),
                "errors": sum(bool(row.get("error")) for row in results), "complete": complete,
                "score": sum(row["scores"]["official"] for row in scored) / expected if complete else None,
                "visual_revisits": sum((row.get("gate") or {}).get("revisit_observed") == "yes" for row in scored),
                "source": "reproduction/gemini-current/episodes/"})
    historical = []
    pilot = read("reproduction/task-memory-benchmark/summary.json")
    if pilot and pilot["complete"]:
        historical.extend({"cohort": "Scripted tasks: 3 cases, seed42", "method": method, "score": value,
                           "source": "reproduction/task-memory-benchmark/summary.json"}
                          for method, value in pilot["local_cohort_means"].items())
    adaptive = read("reproduction/adaptive-four-view/paired-summary.json")
    if adaptive:
        historical.extend({"cohort": "Adaptive GC027: seeds42/43", "method": method,
                           "score": adaptive[f"{method}_mean"], "source": "reproduction/adaptive-four-view/paired-summary.json"}
                          for method in ("stock", "memory"))
    replay = read("reproduction/start-anchor-replay/scores-summary.json")
    if replay and replay["complete"]:
        historical.extend({"cohort": "Fixed GC027 replay: seeds42/43", "method": method, "score": value,
                           "source": "reproduction/start-anchor-replay/scores-summary.json"}
                          for method, value in replay["means"].items())
    moving = read("reproduction/gemini-rejudge-baseline-v6.json")
    moving10 = []
    if moving:
        for arm, key in (("stock caption text, v5 player", "baseline"), ("v6 player (plan progress)", "v6")):
            cases = moving[key]
            scores = {t: c[0] for t, c in cases.items() if c[0] is not None}
            moving10.append({"cohort": "Moving-10 subset (4 GC / 3 IF / 3 out-of-sight), seed 42", "method": arm,
                "scored": len(scores), "expected": len(cases), "complete": len(scores) == len(cases),
                "score": sum(scores.values()) / len(cases) if len(scores) == len(cases) else None,
                "judgements_per_video": 3, "source": "reproduction/gemini-rejudge-baseline-v6.json"})
        insight = moving["baseline_insight"]
        values = [c[0] for c in insight.values() if c[0] is not None]
        moving10.append({"cohort": "Insight-4 (60-second WAIT episodes), seed 42", "method": "stock caption text",
            "scored": len(values), "expected": len(insight), "complete": len(values) == len(insight),
            "score": sum(values) / len(insight) if len(values) == len(insight) else None,
            "judgements_per_video": 3, "source": "reproduction/gemini-rejudge-baseline-v6.json"})
    full = read("reproduction/full-baseline-gemini.json")
    claude_subset = read("reproduction/hillclimb-v6-s42.json")
    claude_cohort = read("reproduction/cohort-stock-v5-s42.json")
    if claude_subset:
        for method, key in (("stock", "baseline_mean"), ("v6 player", "v6_mean")):
            historical.append({"cohort": "Moving-10 subset, seed 42 (Claude judge)", "method": method,
                "score": claude_subset[key], "source": "reproduction/hillclimb-v6-s42.json"})
    if claude_cohort:
        historical.append({"cohort": "Declared 16-episode cohort, seed 42 (Claude judge, incl. 4 insight)", "method": "stock",
            "score": claude_cohort["pooled_mean"], "source": "reproduction/cohort-stock-v5-s42.json"})
    snapshot = {"target": 2.12, "full_benchmark_score": (full or {}).get("overall"), "gemini_moving_subsets": moving10, "full_171_run": full, "published_score_beaten": False,
                "gemini_model": "gemini-3.1-pro-preview", "endpoint": "Vertex AI global",
                "gemini_local_cohorts": cohorts, "historical_claude_cohorts": historical,
                "comparability": "Subset scores are not comparable to published four-dimension overall. Methods only compared within the same cohort/judge. Gemini model alias, player and adapter differences remain documented."}
    (ROOT / "leaderboard.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    lines = ["# WorldSync leaderboard", "",
        "**Goal: beat Genie 3’s published overall 2.12. No comparable full-benchmark score yet.**", "",
        "Compare methods only within the same table/cohort and judge. A higher subset score does not beat the published record.", "",
        "## Published references — not reproduced here", "",
        "| Model | Overall | Scope |", "|---|---:|---|",
        "| Genie 3 |2.12|Published closed-model result|",
        "| LingBot-World2 |1.82|Published hosted-service result; not our local checkpoint measurement|", "",
        "Source: [pinned PlayWorld leaderboard](external/PlayWorld/README.md#leaderboard). Overall averages GC, IF, in-sight evolution and out-of-sight evolution; the full dataset has 171 tasks (48/50/30/43).", "",
        "## Gemini — local cohorts", "",
        "Released prompts, sampling, gate and aggregation; Vertex `gemini-3.1-pro-preview`. The paper's `gemini-3.1-pro` API name is unavailable to our configured API key. Preview-model equivalence is not established. These are GC subsets, not overall scores.", "",
        "| Cohort | Method | Scored | Score | Judge reports revisit |", "|---|---|---:|---:|---:|"]
    for row in cohorts:
        score = f"{row['score']:.2f}" if row["complete"] else "pending"
        lines.append(f"| {row['cohort']} | {row['method']} |{row['scored']}/{row['expected']}|{score}|{row['visual_revisits']}/{row['scored']}|")
    lines.extend(["", "Evidence: [individual Gemini judgments](reproduction/gemini-current/episodes/). Pending/error rows never receive a partial-cohort mean.", "",
        "The earlier [Gemini cohort](reproduction/gemini-rescore.md) found a paired 46/6-cache change of +0.69 over 24 pairs, but only GC/OE coverage, one judge draw per video, generation failures and divergent player trajectories. Its unmatched arm means are not an overall leaderboard.", "",
        ""])
    lines.extend(["## Gemini — mixed-split subsets (3 judgements per video)", "",
        "Same judge and prompts, but cohorts that mix splits and include moving-camera cases. The same video varies by a standard deviation of about 0.33 across judge repeats, and a misread gate can pin a case at 1.0. Differences under about 0.5 on 10 cases are not distinguishable from noise.", "",
        "| Cohort | Method | Scored | Score |", "|---|---|---:|---:|"])
    for row in moving10:
        score = f"{row['score']:.2f}" if row["complete"] else "pending"
        lines.append(f"| {row['cohort']} | {row['method']} |{row['scored']}/{row['expected']}|{score}|")
    lines.extend(["", "**Moving-10: v6 versus stock is +0.10, inside the noise, so not a gain.** v6 also failed 2 of 6 judge-independent [turn checks](reproduction/turn-check-baseline-v6.json) that stock passed 6 of 6. Evidence: [Gemini re-scoring](reproduction/gemini-rejudge-baseline-v6.json).", "",
        "### Full 171-case run (our local checkpoint, current player, caption text)", ""])
    if full and full.get("overall") is not None:
        lines.extend(["| Overall | GC | IF | In-sight | Out-of-sight | Scored |", "|---:|---:|---:|---:|---:|---:|",
            f"|{full['overall']:.2f}|{full['gc']:.2f}|{full['if']:.2f}|{full['insight']:.2f}|{full['outsight']:.2f}|{full['scored']}/171|", ""])
    else:
        lines.extend(["Not scored yet: the 171-case run is in progress. No overall is reported until all four dimensions are covered.", ""])
    lines.extend(["## Claude — historical experiments", "",
        "Retained for audit; do not compare these numbers with Gemini or across different cohorts.", "",
        "| Cohort | Method | Score | Evidence |", "|---|---|---:|---|"])
    for row in historical:
        lines.append(f"| {row['cohort']} | {row['method']} |{row['score']:.2f}|[summary]({row['source']})|")
    lines.extend(["", "The adaptive four-view pilot changed 3.60→3.90, but neither seed visibly returned; the gain is inconclusive. The fixed start-anchor follow-up also failed all visual revisits. [Identical-video judge variability](reproduction/judge-repeatability.json) reached 2.70→3.75. GC013 is quarantined for an upstream image/caption mismatch.", "",
        "## Other measured results", "",
        "- [Turn check](reproduction/turn-check-baseline-v6.json): judge-independent yaw-from-frames check; stock passed 6/6 turn tasks, v6 4/6. Gemini failed OE057 three times for both arms although the video turned as instructed.",
        "- [Protocol finding](reproduction/text-prompt-finding.md): the released PlayWorld harness submits `task.prompt` as the world-model text, while our adapter sent the static image caption. `--text-prompt prompt` exists; its test is incomplete (GPU shared, one case out-of-memory), so no score is claimed.",
        "- [Long-return memory](reproduction/memory-validation/README.md): a 31.8-second episode with 24 seconds away retained the landmark; residual framing/text drift remains. Not a benchmark score.",
        "- [Local submaps](reproduction/spatial-local-submaps/summary.json): better mall localization, bridge and drifted-stock failures; not generator conditioning.",
        "- [Persistent runner](reproduction/modal-runner.md): 3.6x steady short-job throughput in the reported seven-job trial, exact outputs on the same GPU model. [Mixed-scene batch](reproduction/lingbot-batch-validation/) preserves all 42 chunk hashes.",
        ""])
    lines.extend(extra_sections())
    lines.extend(["", "Regenerate: `.venv-lingbot/bin/python scripts/update_leaderboard.py`. Machine-readable snapshot: [leaderboard.json](leaderboard.json).", ""])
    (ROOT / "LEADERBOARD.md").write_text("\n".join(lines))


if __name__ == "__main__":
    generate()
