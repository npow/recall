# WorldSync leaderboard

**Goal: beat Genie 3’s published overall 2.12. No comparable full-benchmark score yet.**

Compare methods only within the same table/cohort and judge. A higher subset score does not beat the published record.

## Published references — not reproduced here

| Model | Overall | Scope |
|---|---:|---|
| Genie 3 |2.12|Published closed-model result|
| LingBot-World2 |1.82|Published hosted-service result; not our local checkpoint measurement|

Source: [pinned PlayWorld leaderboard](external/PlayWorld/README.md#leaderboard). Overall averages GC, IF, in-sight evolution and out-of-sight evolution; the full dataset has 171 tasks (48/50/30/43).

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

The adaptive four-view pilot changed 3.60→3.90, but neither seed visibly returned; the gain is inconclusive. The fixed start-anchor follow-up also failed all visual revisits. [Identical-video judge variability](reproduction/judge-repeatability.json) reached 2.70→3.75. GC013 is quarantined for an upstream image/caption mismatch.

## Other measured results

- [Long-return memory](reproduction/memory-validation/README.md): a 31.8-second episode with 24 seconds away retained the landmark; residual framing/text drift remains. Not a benchmark score.
- [Local submaps](reproduction/spatial-local-submaps/summary.json): better mall localization, bridge and drifted-stock failures; not generator conditioning.
- [Persistent runner](reproduction/modal-runner.md): 3.6x steady short-job throughput in the reported seven-job trial, exact outputs on the same GPU model. [Mixed-scene batch](reproduction/lingbot-batch-validation/) preserves all 42 chunk hashes.

Regenerate: `.venv-lingbot/bin/python scripts/update_leaderboard.py`. Machine-readable snapshot: [leaderboard.json](leaderboard.json).
