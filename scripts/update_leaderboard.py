"""Regenerate the repository leaderboard from complete, separate cohorts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    path = ROOT / name
    return json.loads(path.read_text()) if path.exists() else None


def powered_rows():
    """Seed-42 paired Gemini comparison on 14 GC/OE development cases (stock vs budget memory vs 46/6)."""
    data = read("reproduction/powered-comparison/pooled/powered-analysis.json")
    expected = 84
    if not data:
        data, expected = read("reproduction/powered-comparison/seed42/powered-analysis.json"), 42
    if not data or data.get("judge_errors") or data["episodes_scored"] != expected:
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
    recovered = read("reproduction/full1-recovered-judging/summary.json")
    if recovered:
        lines.extend(["", "## Full171 stock versus budget run", "",
            "All342 generation records are verified complete. Judging resumes from preserved video-hash-verified reports after the original scorer lost credentials. Overall scores remain pending until all171 cases in each arm are scored.", "",
            "| Arm | Generated | Judged | Overall |", "|---|---:|---:|---:|"])
        for arm, result in recovered["arms"].items():
            count = sum(d["scored"] for d in result["dimensions"].values())
            overall = result["overall_equal_dimension_mean"]
            value = f"{overall:.4f}" if overall is not None else "pending"
            lines.append(f"| {arm} |171/171|{count}/171|{value}|")
        lines.extend(["", "[Generation plan, completion records and judgments](reproduction/full1-recovered-judging/README.md). Uses the integrated checkpoint, streamed Haiku player and Gemini preview alias; hosted-service equivalence is not established. The released rubric admits partial returns, so an overall score alone does not prove revisit memory."])
    full_history = read("reproduction/full-history-GC010/gemini-summary.json")
    if full_history:
        lines.extend(["", "## Matched14B full-history diagnostic", "",
            "One scripted GC010 seed42 case, same H200/SDPA/seed/controls. Stock18, four-view retrieval and native58 match raw pixels through chunk3; the latter retains the whole56-latent trajectory.", "",
            "| Arm | Released GC rubric | Judge reports revisit |", "|---|---:|---|"])
        for arm, result in full_history["results"].items():
            lines.append(f"| {arm} |{result['score']:.2f}|{result['judge_reports_revisit']}|")
        lines.extend(["", "[Videos, frame comparison and frozen inputs](reproduction/full-history-GC010/README.md). Native58 and retrieval recover recognizable initial street buildings; framing and details still drift. This is a single development diagnostic, not an overall score or proof of exact geometry. Native58 peaks at121.39GB allocated. The NanoKV identity adapter reproduces all14 native58 chunks exactly on H200; lossy compressed18/58 tests are running, with no quality gain claimed yet."])
    powered = powered_rows()
    if powered:
        lines.extend(["", "## Gemini — paired GC/OE development cohort (seeds 42 and 43)", "",
            "14 GC and out-of-sight development cases (7 GC, 7 out-of-sight), three arms, two seeds, same cases. The cases were not hash-selected: 7 come from the first development cohort and 7 (GC005, OE044, GC015, OE056, OE055, GC025, GC019) were chosen for long reference-plan duration, before outcomes. Every arm ran with `--end-guard` and player retries, which the full run does not use. "
            "Scores are the Gemini mean over GC and out-of-sight episodes only, **not** the four-dimension overall, "
            "so they are **not comparable with 2.12 or 1.82**. Paired delta is against stock over (case, seed) pairs; "
            "wins/ties/losses in brackets (28 pairs). No arm is established better than stock overall. The one interval that clears zero, 46/6 on out-of-sight (+0.84 [+0.20, +1.62]), is one of eight uncorrected contrasts, comes mostly from seed 43, and Opus sees nothing (+0.05), so it is a lead, not a result.", "",
            "| Arm | n | Gemini mean [95% CI] | GC | Out-of-sight | Gate non-fail | Paired Δ vs stock [95% CI] (W/T/L) |",
            "|---|---:|---|---:|---:|---:|---|", *powered, "",
            "Evidence: [analysis](reproduction/powered-comparison/pooled/powered-analysis.json), "
            "[case-clustered CIs](reproduction/powered-comparison/pooled/case-cluster-ci.json), "
            "[full tables and limits](reproduction/powered-comparison/pooled/README.md), [protocol](reproduction/powered-comparison.md)."])
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
        "The pass criterion for beating 2.12 was fixed in advance in "
        "[pass-criterion.md](reproduction/pass-criterion.md): equal mean of four dimension means, released Gemini scorer, "
        "identical-harness stock baseline, full 171 cases, lower CI bound above 2.12.", "",
        "A full 171-case run is complete; preliminary tally stock about 1.84, budget memory about 1.80 (no gain); official table pending. "
        "**The pass criterion is not met.** An exploratory check finds memory did fire in 136 of 171 budget episodes, the earlier +0.32 does not "
        "replicate on its own 14 cases inside the full run (+0.04), and the judge scored 5 of 28 identical-pixel video pairs differently "
        "(by up to 4.0 points), so single-call differences under about 0.3 are noise: "
        "[did memory act?](reproduction/full-run-memory-fired.md)."])
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
    full_claude = read("reproduction/full-baseline-claude.json")
    claude_subset = read("reproduction/hillclimb-v6-s42.json")
    claude_cohort = read("reproduction/cohort-stock-v5-s42.json")
    if claude_subset:
        for method, key in (("stock", "baseline_mean"), ("v6 player", "v6_mean")):
            historical.append({"cohort": "Moving-10 subset, seed 42 (Claude judge)", "method": method,
                "score": claude_subset[key], "source": "reproduction/hillclimb-v6-s42.json"})
    if claude_cohort:
        historical.append({"cohort": "Declared 16-episode cohort, seed 42 (Claude judge, incl. 4 insight)", "method": "stock",
            "score": claude_cohort["pooled_mean"], "source": "reproduction/cohort-stock-v5-s42.json"})
    text_confirmation = read("reproduction/task-text-confirmation/summary.json")
    reader_fit = read("reproduction/learned-memory-reader/real-fit-summary.json")
    small_context = read("reproduction/lingbot-small/gemini-summary.json")
    text_pilot = read("reproduction/task-text-pilot/summary.json")
    snapshot = {"full_history_diagnostic": read("reproduction/full-history-GC010/gemini-summary.json"), "full1_recovered": read("reproduction/full1-recovered-judging/summary.json"), "task_text_confirmation": text_confirmation, "learned_reader_surrogate": reader_fit, "small_model_context_diagnostic": small_context, "task_text_development": text_pilot, "target": 2.12, "full_benchmark_score": (full or {}).get("overall"), "gemini_moving_subsets": moving10, "full_171_run": full, "published_score_beaten": False,
                "gemini_model": "gemini-3.1-pro-preview", "endpoint": "Vertex AI global",
                "gemini_local_cohorts": cohorts, "historical_claude_cohorts": historical,
                "comparability": "Subset scores are not comparable to published four-dimension overall. Methods only compared within the same cohort/judge. Gemini model alias, player and adapter differences remain documented."}
    (ROOT / "leaderboard.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    lines = ["# WorldSync leaderboard", "",
        "**Goal: beat Genie 3’s published overall 2.12. No comparable full-benchmark score yet.**", "",
        "Compare methods only within the same table/cohort and judge. A higher subset score does not beat the published record.", "", "Method and evidence: [METHOD.md](METHOD.md).", "",
        "## Published references — not reproduced here", "",
        "| Model | Overall | Scope |", "|---|---:|---|",
        "| Genie 3 |2.12|Published closed-model result|",
        "| LingBot-World2 |1.82|Published hosted-service result; not our local checkpoint measurement|", "",
        "Source: [pinned PlayWorld leaderboard](https://github.com/kxding/PlayWorld/blob/eeaea12e050a90b815d11d030e943e7b3181da21/README.md#leaderboard). Overall averages GC, IF, in-sight evolution and out-of-sight evolution; the full dataset has 171 tasks (48/50/30/43).", "",
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
        lines.extend(["No completed 171-case result is available here. Partial runs do not establish an overall; all declared cases and all four dimensions are required.", ""])
    if full_claude and full_claude.get("overall") is not None:
        lines.extend(["Claude-judged version of the same 171 videos (different judge, not comparable to the published table):", "",
            "| Overall | GC | IF | In-sight | Out-of-sight | Scored |", "|---:|---:|---:|---:|---:|---:|",
            f"|{full_claude['overall']:.2f}|{full_claude['gc']:.2f}|{full_claude['if']:.2f}|{full_claude['insight']:.2f}|{full_claude['outsight']:.2f}|{full_claude['scored']}/171|", ""])
    lines.extend(["## Claude — historical experiments", "",
        "Retained for audit; do not compare these numbers with Gemini or across different cohorts.", "",
        "| Cohort | Method | Score | Evidence |", "|---|---|---:|---|"])
    for row in historical:
        lines.append(f"| {row['cohort']} | {row['method']} |{row['score']:.2f}|[summary]({row['source']})|")
    lines.extend(["", "The adaptive four-view pilot changed 3.60→3.90, but neither seed visibly returned; the gain is inconclusive. The fixed start-anchor follow-up also failed all visual revisits. [Identical-video judge variability](reproduction/judge-repeatability.json) reached 2.70→3.75. GC013 is quarantined for an upstream image/caption mismatch.", "",
        "## Other measured results", "",
        "- [Rotation and yaw-clamp test](reproduction/rotation-clamp-test.json): 4 GC 360-degree rotation cases scored 2.22 at 60 deg/s, 2.19 at 50 deg/s and 1.97 with the plan clamp; 3 OE turn-and-return cases rose 1.19 to 1.83 with the clamp. One run per case, inside noise; the trajectory gate already passed at 60 deg/s, so turn overshoot is not the limiting factor. The bounded pose-memory arm on the same cases also showed no gain (2.60 vs 2.44 on 3 GC cases). No claimed improvement.",
        "- [Turn check](reproduction/turn-check-baseline-v6.json): judge-independent yaw-from-frames check; stock passed 6/6 turn tasks, v6 4/6. Gemini failed OE057 three times for both arms although the video turned as instructed.",
        "- [Protocol finding](reproduction/text-prompt-finding.md): the released PlayWorld harness submits `task.prompt` as the world-model text, while our adapter sent the static image caption. `--text-prompt prompt` exists; the completed four-case development comparison and its limitations are reported above; held-out confirmation is pending.",
        "- [Long-return memory](reproduction/memory-validation/README.md): a 31.8-second episode with 24 seconds away retained the landmark; residual framing/text drift remains. Not a benchmark score.",
        "- [Local submaps](reproduction/spatial-local-submaps/summary.json): better mall localization, bridge and drifted-stock failures; not generator conditioning.",
        "- [Persistent runner](reproduction/modal-runner.md): 3.6x steady short-job throughput in the reported seven-job trial, exact outputs on the same GPU model. [Mixed-scene batch](reproduction/lingbot-batch-validation/) preserves all 42 chunk hashes.",
        ""])
    if text_pilot and text_pilot.get("complete"):
        lines.extend(["## Gemini — IF text-conditioning development pilot", "",
            "Four predeclared IF cases, seed 42; 12 completed adaptive episodes. Same H200, native 18/6 cache and player. These are development results, not an overall score.", "",
            "| Text conditioning | Mean |", "|---|---:|"])
        for mode, score in text_pilot["means"].items():
            lines.append(f"| {mode} |{score:.2f}|")
        lines.extend(["", "Caption plus objective is awaiting separate case/seed confirmation. Physical failures remain: walking through glass and disappearing carried objects. [Protocol, videos and per-case scores](reproduction/task-text-pilot/README.md).", ""])
    if small_context and small_context.get("complete"):
        lines.extend(["## Gemini — small-model scripted context diagnostic", "",
            "Official small checkpoint, GC027, seed42, thirteen native fixed-camera chunks. One judge draw per video; no adaptive player. This does not establish a benchmark improvement.", "",
            "| Cache | Released GC rubric (partial admitted) | Revisit observed |", "|---|---:|---|"])
        for arm, row in small_context["results"].items():
            lines.append(f"| {arm} |{row['score']:.2f}|{row['gate']['revisit_observed']}|")
        lines.extend(["", "Both videos fail the requested return. [Scoring audit](reproduction/lingbot-small/scoring-audit.md): partial trajectories pass the released gate, so visual rubric scores can remain high without a revisit. [Videos, frame comparison and caveats](reproduction/lingbot-small/README.md).", ""])
    if text_confirmation:
        lines.extend(["## Gemini — frozen task-text confirmation", "",
            "All16 episodes generated. Partition means appear only when every declared episode in that partition has a judgment or retained generation failure.", "",
            "| Partition | Judged or generation failed | Caption | Caption + objective |", "|---|---:|---:|---:|"])
        for partition, row in text_confirmation.get("partition_reports", {}).items():
            means = row.get("means")
            caption = f"{means['caption']:.2f}" if means else "pending"
            both = f"{means['both']:.2f}" if means else "pending"
            lines.append(f"| {partition} |{row['scored_or_generation_failed']}/{row['expected_episodes']}|{caption}|{both}|")
        lines.extend(["", "The development gain did not reproduce on the new seed; the frozen promotion condition failed. One held-out judgment is capacity-limited. [Protocol, videos and recovery](reproduction/task-text-confirmation/README.md).", ""])
    if reader_fit and reader_fit.get("complete"):
        lines.extend(["## Learned reader — denoising surrogate, not PlayWorld", "",
            f"The60-update real-data fit passed frozen-base and checkpoint-reload checks, but correct memory beats both controls on only{reader_fit['heldout_cases_passing']}/{reader_fit['heldout_cases']} held-out sources. The predeclared surrogate gate failed; the adapter is not promoted. [Raw results and video diagnostic](reproduction/learned-memory-reader/README.md).", ""])
    lines.extend(extra_sections())
    lines.extend(["", "Regenerate: `.venv-lingbot/bin/python scripts/update_leaderboard.py`. Machine-readable snapshot: [leaderboard.json](leaderboard.json).", ""])
    (ROOT / "LEADERBOARD.md").write_text("\n".join(lines))


if __name__ == "__main__":
    generate()
