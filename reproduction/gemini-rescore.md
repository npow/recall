# Gemini rescoring of existing episodes

The user lifted the no-Gemini rule for final scoring only and asked for their `gcloud`
login. [`worldsync/gemini_vertex_judge.py`](../worldsync/gemini_vertex_judge.py) runs
PlayWorld's released Rubric Verifier through Vertex AI (model `gemini-3.1-pro-preview`,
global endpoint). Prompts, sampling, rubric, gate and aggregation are the released code;
only the endpoint, credentials and model id differ (see the deviation entry in
[lingbot.md](lingbot.md)). Judge-only change: the episodes, player (Claude Haiku 4.5),
controller and checkpoint are the ones already generated.

**Not a leaderboard reproduction.** One Gemini sample per episode (temperature 0.1), a
15-case development subset, GC and out-of-sight OE only, and a different player and
adapter from the paper.

## Coverage

* 85 episodes attempted (84 guarded cohort directories plus the GC007 v3 episode); 10 had
  no video because generation failed, so **75 were scored**: stock 29, 46/6 25,
  feedback 10, feedback+turn-compensation 10, GC007 v3 1. Generation failures are excluded,
  not counted as 1.0 (the released aggregator would count absent tasks as 1.0).
* Cases: GC005, 007, 015, 017, 019, 024, 025, 027 and out-of-sight OE026, 029, 044, 055,
  056, 057, 069, two seeds (42, 43). **IF and in-sight OE were not run**, so there is no
  four-dimension overall.
* One transient `RemoteProtocolError` was retried and succeeded. About 2.55 M tokens in
  total across about 135 calls.

## Gemini vs our Opus judge (60 episodes scored by both)

| Measure | Value |
| --- | --- |
| Pearson / Spearman of per-episode score | 0.31 / 0.38 |
| Mean bias, Gemini − Opus | −0.09 |
| Gate fail vs non-fail agreement | 49/60 (82%) |
| Exact gate verdict agreement | 34/60 (57%) |
| Gate non-fail rate, Gemini / Opus | 88% / 90% |

The judges agree weakly on individual episodes. Gemini passed 12 episodes Opus
called partial and failed 6 that Opus did not fail. The GC007 v3 episode got
**5.0 from Gemini against 3.2 from Opus**, with Gemini calling a visibly garbled-sign
shopfront run "perfect consistency" and the revisit observed. Gemini appears lenient on
low-motion videos (GC025 also scores 5.0 in both arms).

## Per-arm means (Gemini, released aggregation)

| Arm | n | GC (n) | Out-of-sight OE (n) | 2-dim mean | Gate non-fail | Opus mean, same episodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stock | 29 | 2.40 (16) | 1.18 (13) | **1.79** | 24/29 | 2.30 |
| 46/6 cache | 25 | 3.21 (14) | 1.56 (11) | **2.39** | 21/25 | 2.27 |
| feedback | 10 | 2.28 (10) | n/a | 2.28 | 10/10 | 2.96 |
| feedback + turn comp. | 10 | 2.14 (10) | n/a | 2.14 | 8/10 | 2.57 |
| GC007 v3 stock | 1 | 5.00 | n/a | 5.00 | 1/1 | 3.20 |

Arms cover different case sets, so compare paired rows (same case and seed) below, not these
means. Per-case values are in [tables.md](gemini-rescore/tables.md) and
[per-episode.csv](gemini-rescore/per-episode.csv).

## Paired comparison against stock (same case and seed)

| Arm | Pairs | Mean Δ (Gemini) | 95% bootstrap CI | W / T / L | Sign test p | Opus Δ, same pairs |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| 46/6 | 24 | **+0.69** (GC +0.90, OE +0.38) | [+0.19, +1.23] | 12 / 9 / 3 | 0.035 | +0.06 |
| feedback | 10 | +0.26 | [−0.92, +1.28] | 4 / 5 / 1 | 0.375 | +0.75 |
| feedback + comp. | 10 | +0.18 | [−0.52, +0.94] | 2 / 6 / 2 | 1.0 | −0.38 |

* **46/6 looks better under Gemini and not under Opus.** The Gemini gain is concentrated
  in a few cases (GC024 +1.7, GC027 +1.6, OE069 +1.6, GC007 +1.0, GC019 +1.0). It is
  suggestive, not established: the two arms' player trajectories diverge after the first
  chunk, the sign test treats pairs as independent, the judges disagree on direction for
  feedback, and there is one Gemini sample per episode, so judge sampling noise is unmeasured.
* **Feedback arms show no reliable Gemini effect** (CIs span zero); Opus and Gemini disagree
  on their sign.

## Per-dimension means (Gemini rubric, scored only when the gate did not fail)

GC
| Arm | n | identity_id | identity_texture | color | relative_position |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock | 15 | 2.20 | 2.53 | 3.20 | 2.33 |
| 46/6 | 13 | 3.00 | 3.62 | 4.31 | 3.00 |
| feedback | 10 | 2.00 | 2.30 | 3.20 | 1.90 |
| feedback + comp. | 8 | 2.50 | 2.25 | 2.38 | 2.50 |

Out-of-sight OE (stock n=7 to 9, 46/6 n=6 to 8 per category; others n=2)
| Dimension | stock | 46/6 |
| --- | ---: | ---: |
| hidden_state_progression | 1.29 | 2.00 |
| process_accumulation | 1.14 | 2.00 |
| revealed_state_difference | 1.43 | 2.00 |
| subject_identity | 1.11 | 1.88 |

The gains appear in identity, texture, color and position for GC and in hidden-state
categories for OE. Categories with n=2 (for example temporal_scene_consistency) are too few to read.
Gate counts (Gemini): stock 5 fail / 9 partial / 15 pass; 46/6 4 / 10 / 11;
feedback 0 / 9 / 1; feedback+comp. 2 / 6 / 2.

## Does stock land near the paper's scale?

| | GC | Out-of-sight OE | Mean of the two |
| --- | ---: | ---: | ---: |
| Paper, LingBot-World-Infinity | 2.04 | 1.16 | 1.60 |
| Ours, stock, Gemini | 2.40 | 1.18 | 1.79 |

Out-of-sight OE matches (1.18 vs 1.16). GC is 0.36 higher and the two-dimension mean 0.19
higher. The paper's overall 1.82 also averages IF (2.13) and in-sight OE (1.95), which we did
not run, so 1.79 is **not** comparable to 1.82; the scale is plausible, not verified. Our 15
development cases are tilted toward long-absence tasks, the player and adapter differ, and
the local checkpoint is not confirmed to match the hosted service.

## Caveats

* Vertex `gemini-3.1-pro-preview` is not confirmed to be the paper's judge checkpoint.
* One sample per episode; no repeat-call variance. A second pass on the cases that drive the
  46/6 gain would be the cheapest check.
* Players are stochastic, so stock and 46/6 pairs are different rollouts, not the same video
  under two caches.
* Videos, tokens and per-call rationales are not committed; `gemini-results.json` holds scores,
  gate verdicts and category scores only.
