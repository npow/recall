# WorldSync leaderboard

**Goal: beat Genie 3’s published overall 2.12. No comparable full-benchmark score yet.**

Compare methods only within the same table/cohort and judge. A higher subset score does not beat the published record.

Method and evidence: [METHOD.md](METHOD.md).

## Published references — not reproduced here

| Model | Overall | Scope |
|---|---:|---|
| Genie 3 |2.12|Published closed-model result|
| LingBot-World2 |1.82|Published hosted-service result; not our local checkpoint measurement|

Source: [pinned PlayWorld leaderboard](https://github.com/kxding/PlayWorld/blob/eeaea12e050a90b815d11d030e943e7b3181da21/README.md#leaderboard). Overall averages GC, IF, in-sight evolution and out-of-sight evolution; the full dataset has 171 tasks (48/50/30/43).

## Gemini — local cohorts

Released prompts, sampling, gate and aggregation; Vertex `gemini-3.1-pro-preview`. The paper's `gemini-3.1-pro` API name is unavailable to our configured API key. Preview-model equivalence is not established. These are GC subsets, not overall scores.

| Cohort | Method | Scored | Score | Judge reports revisit |
|---|---|---:|---:|---:|
| Scripted task pilot | stock |3/3|1.00|3/3|
| Scripted task pilot | four-view |3/3|3.53|3/3|
| Adaptive GC027 | stock |2/2|3.10|0/2|
| Adaptive GC027 | memory |2/2|1.50|1/2|
| Start-anchor replay | stock |2/2|1.70|0/2|
| Start-anchor replay | four-view |2/2|1.80|0/2|
| Start-anchor replay | start-anchor |2/2|3.00|0/2|

Evidence: [individual Gemini judgments](reproduction/gemini-current/episodes/). Pending/error rows never receive a partial-cohort mean.

The earlier [Gemini cohort](reproduction/gemini-rescore.md) found a paired 46/6-cache change of +0.69 over 24 pairs, but only GC/OE coverage, one judge draw per video, generation failures and divergent player trajectories. Its unmatched arm means are not an overall leaderboard.


## Gemini — mixed-split subsets (3 judgements per video)

Same judge and prompts, but cohorts that mix splits and include moving-camera cases. The same video varies by a standard deviation of about 0.33 across judge repeats, and a misread gate can pin a case at 1.0. Differences under about 0.5 on 10 cases are not distinguishable from noise.

| Cohort | Method | Scored | Score |
|---|---|---:|---:|
| Moving-10 subset (4 GC / 3 IF / 3 out-of-sight), seed 42 | stock caption text, v5 player |10/10|1.64|
| Moving-10 subset (4 GC / 3 IF / 3 out-of-sight), seed 42 | v6 player (plan progress) |10/10|1.74|
| Insight-4 (60-second WAIT episodes), seed 42 | stock caption text |4/4|1.45|

**Moving-10: v6 versus stock is +0.10, inside the noise, so not a gain.** v6 also failed 2 of 6 judge-independent [turn checks](reproduction/turn-check-baseline-v6.json) that stock passed 6 of 6. Evidence: [Gemini re-scoring](reproduction/gemini-rejudge-baseline-v6.json).

### Full 171-case run (our local checkpoint, current player, caption text)

No completed 171-case result is available here. Partial runs do not establish an overall; all declared cases and all four dimensions are required.

## Claude — historical experiments

Retained for audit; do not compare these numbers with Gemini or across different cohorts.

| Cohort | Method | Score | Evidence |
|---|---|---:|---|
| Scripted tasks: 3 cases, seed42 | stock |2.47|[summary](reproduction/task-memory-benchmark/summary.json)|
| Scripted tasks: 3 cases, seed42 | retrieval |3.57|[summary](reproduction/task-memory-benchmark/summary.json)|
| Adaptive GC027: seeds42/43 | stock |3.60|[summary](reproduction/adaptive-four-view/paired-summary.json)|
| Adaptive GC027: seeds42/43 | memory |3.90|[summary](reproduction/adaptive-four-view/paired-summary.json)|
| Fixed GC027 replay: seeds42/43 | stock |3.80|[summary](reproduction/start-anchor-replay/scores-summary.json)|
| Fixed GC027 replay: seeds42/43 | four-view |4.00|[summary](reproduction/start-anchor-replay/scores-summary.json)|
| Fixed GC027 replay: seeds42/43 | start-anchor |3.75|[summary](reproduction/start-anchor-replay/scores-summary.json)|
| Moving-10 subset, seed 42 (Claude judge) | stock |2.22|[summary](reproduction/hillclimb-v6-s42.json)|
| Moving-10 subset, seed 42 (Claude judge) | v6 player |2.67|[summary](reproduction/hillclimb-v6-s42.json)|
| Declared 16-episode cohort, seed 42 (Claude judge, incl. 4 insight) | stock |2.17|[summary](reproduction/cohort-stock-v5-s42.json)|

The adaptive four-view pilot changed 3.60→3.90, but neither seed visibly returned; the gain is inconclusive. The fixed start-anchor follow-up also failed all visual revisits. [Identical-video judge variability](reproduction/judge-repeatability.json) reached 2.70→3.75. GC013 is quarantined for an upstream image/caption mismatch.

## Other measured results

- [Turn check](reproduction/turn-check-baseline-v6.json): judge-independent yaw-from-frames check; stock passed 6/6 turn tasks, v6 4/6. Gemini failed OE057 three times for both arms although the video turned as instructed.
- [Protocol finding](reproduction/text-prompt-finding.md): the released PlayWorld harness submits `task.prompt` as the world-model text, while our adapter sent the static image caption. `--text-prompt prompt` exists; the completed four-case development comparison and its limitations are reported above; held-out confirmation is pending.
- [Long-return memory](reproduction/memory-validation/README.md): a 31.8-second episode with 24 seconds away retained the landmark; residual framing/text drift remains. Not a benchmark score.
- [Local submaps](reproduction/spatial-local-submaps/summary.json): better mall localization, bridge and drifted-stock failures; not generator conditioning.
- [Persistent runner](reproduction/modal-runner.md): 3.6x steady short-job throughput in the reported seven-job trial, exact outputs on the same GPU model. [Mixed-scene batch](reproduction/lingbot-batch-validation/) preserves all 42 chunk hashes.

## Gemini — IF text-conditioning development pilot

Four predeclared IF cases, seed 42; 12 completed adaptive episodes. Same H200, native 18/6 cache and player. These are development results, not an overall score.

| Text conditioning | Mean |
|---|---:|
| caption |1.00|
| prompt |1.25|
| both |1.71|

Caption plus objective is awaiting separate case/seed confirmation. Physical failures remain: walking through glass and disappearing carried objects. [Protocol, videos and per-case scores](reproduction/task-text-pilot/README.md).


## Gemini — paired GC/OE development cohort (seeds 42 and 43)

14 hash-selected GC and out-of-sight cases (7 GC, 7 out-of-sight), three arms, two seeds, same cases, `--end-guard` and player retries on every arm. Scores are the Gemini mean over GC and out-of-sight episodes only, **not** the four-dimension overall, so they are **not comparable with 2.12 or 1.82**. Paired delta is against stock over (case, seed) pairs; wins/ties/losses in brackets (28 pairs). No arm is established better than stock overall. The one interval that clears zero, 46/6 on out-of-sight (+0.84 [+0.20, +1.62]), is one of eight uncorrected contrasts, comes mostly from seed 43, and Opus sees nothing (+0.05), so it is a lead, not a result.

| Arm | n | Gemini mean [95% CI] | GC | Out-of-sight | Gate non-fail | Paired Δ vs stock [95% CI] (W/T/L) |
|---|---:|---|---:|---:|---:|---|
| stock 18/6 |28|1.81 [1.35, 2.35]|2.46|1.16|21/28|baseline|
| budget memory (K10 native) |28|2.13 [1.66, 2.67]|2.81|1.45|25/28|+0.32 [-0.08, +0.75] (9/14/5)|
| 46/6 cache |28|2.26 [1.76, 2.83]|2.53|2.00|21/28|+0.46 [-0.22, +1.15] (12/10/6)|

Evidence: [analysis](reproduction/powered-comparison/pooled/powered-analysis.json), [case-clustered CIs](reproduction/powered-comparison/pooled/case-cluster-ci.json), [full tables and limits](reproduction/powered-comparison/pooled/README.md), [protocol](reproduction/powered-comparison.md).

## Gemini — earlier rescoring of the first cohorts

75 episodes from the first stock / 46/6 / feedback cohorts, rescored with the same Gemini judge (GC and out-of-sight only; 10 generation failures excluded; different player rollouts per arm). A paired 46/6 change of +0.69 [+0.19, +1.23] over 24 pairs was found here, but the powered cohort above found +0.05 on different episodes, so the effect is **not established**.

| Arm | n | GC (n) | Out-of-sight (n) | GC+OE mean | Gate non-fail |
|---|---:|---:|---:|---:|---:|
| stock 18/6 |29|2.40 (16)|1.18 (13)|1.79|24/29|
| 46/6 cache |25|3.21 (14)|1.56 (11)|2.39|21/25|
| measured-motion feedback |10|2.28 (10)|n/a (0)|2.28|10/10|
| feedback + turn compensation |10|2.14 (10)|n/a (0)|2.14|8/10|

Evidence: [report](reproduction/gemini-rescore.md), [per-episode scores](reproduction/gemini-rescore/per-episode.csv).

## Fixed-path revisit diagnostics (DINO, not a benchmark score)

Scripted camera paths with a guaranteed return to the start view; metric is DINO similarity of the final frame to the start view. Single seeds, crude proxy, two scenes. They show *where* memory helps, not a PlayWorld gain. On the long 24-chunk absence (GC007): stock about 0.30-0.40, bounded pose memory about 0.77, 46/6 cache about 0.89 (115 GB). Negative results kept: memory LoRA (degrades ordinary generation), re-anchoring, DeCoPrune ports, prompt/translation tricks, measured-motion feedback (within noise). See [integration summary](reproduction/INTEGRATION.md), [revisit matrix](reproduction/revisit/README.md) and [budget memory](reproduction/budgetmem/README.md).

## What would change the headline

Nothing here is a full-benchmark score. The pass criterion for beating 2.12 was fixed in advance in [pass-criterion.md](reproduction/pass-criterion.md): equal mean of four dimension means, released Gemini scorer, identical-harness stock baseline, full 171 cases, lower CI bound above 2.12.

Regenerate: `.venv-lingbot/bin/python scripts/update_leaderboard.py`. Machine-readable snapshot: [leaderboard.json](leaderboard.json).
