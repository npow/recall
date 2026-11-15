# Adaptive episodes: stock 18/6 vs experimental 46/6 cache

Same Haiku reference-guided player, same pinned local Claude rubric (Opus judge) as the GC007 v3
episode, seed 42, one H200 per episode. The 46/6 arm uses `--experimental-adaptive-ablation`
(labelled in run.json) and is NOT a stock or reproduction claim. Local judge: not comparable with
the paper's Gemini scores (1.82 / 2.12). One seed.

| Case | Stock 18/6 | 46/6 (experimental) | Stock secs / actions | 46/6 secs / actions |
| --- | --- | --- | --- | --- |
| GC007 | 1.0 | 1.0 | 18.8s / 20 | 16.8s / 18 |
| GC024 | 2.6 | 2.7 | 5.8s / 7 | 6.8s / 8 |
| GC017 | n/a (generation failed or produced no artifacts) | 3.0 | - | 10.8s / 12 |
| GC027 | 2.2 | 2.2 | 7.8s / 9 | 19.8s / 21 |
| OE026 | 1.75 | 1.25 | 5.8s / 7 | 5.8s / 7 |
| OE029 | 1.5 | n/a (judge failed) | 4.8s / 6 | 4.8s / 6 |
| OE057 | 1.75 | 1.75 | 3.8s / 5 | 3.8s / 5 |
| OE069 | 1.75 | 2.0 | 5.8s / 7 | 5.8s / 7 |

Paired mean over the 6 cases where both arms scored: stock 1.84, 46/6 1.82 (difference -0.03, noise).
Unpaired: GC017 46/6 = 3.0 (stock generation failed), OE029 stock = 1.5 (46/6 judge call failed).

## Why this does not test the long-horizon claim
Every episode ended early: the player issued END after 5 to 21 actions, producing only 3.8 to 19.8
seconds of video (budget was 40 interactions). The stock window is 18 latent frames (about 4.5 s),
so most episodes barely exceed it and a larger cache has little to retrieve. The fixed-path revisit
matrix (reproduction/revisit/) shows the cache effect on long absences; adaptive episodes do not
reach that regime. Player ending early and false success claims (see GC007: judge gate fail,
instruction following 0) dominate the score, as in GC007 v3.

Raw results: cohort-results.json. Videos and run.json are in Modal volume worldsync-lingbot-results
under `cohort-*` (nissan-pow).

## Guarded rerun (two seeds, 15 development cases)

`--end-guard` on for both arms, seeds 42 and 43, 8 original cases plus 7 long-absence cases chosen by
reference-plan duration only. Full table: `guarded-results.md` / `guarded-results.json`
(`scripts/aggregate_cohort.py`). Diagnosis of the early ENDs: `README-early-end.md`.

* Paired scores (46/6 minus stock), 24 pairs where both scored: mean +0.09, 9 wins, 10 ties, 5 losses.
  Per-case seed variance (up to 2 points, e.g. GC015 3.4 vs 2.1 on stock) is as large as this effect.
  Long-absence pairs (commanded revisit absence over 4.5 s, n=5): mean -0.16.
* Mean score: stock 2.23 (n=29), 46/6 2.19 (n=25). Episodes with a commanded-path revisit: 18/30 per arm.
* 6 of 60 episodes failed (3 generation, 3 judge), so arms are unbalanced.
* Local Opus judge, not comparable with the paper's 1.82 / 2.12.
