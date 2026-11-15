# Powered paired judged comparison: stock vs bounded pose memory vs 46/6 cache

Two seeds (42, 43), 14 development cases (7 GC, 7 out-of-sight OE; long-absence cases chosen by reference-plan
duration, outcome-independent), three arms, adaptive Haiku player with `--end-guard` and `--player-retries 2` on
every arm. 84 episodes: 0 generation failures, 2 Opus judge infrastructure failures (stock GC015 s43, GC017 s43;
their videos were still scored by Gemini), 0 Gemini errors. Modal `nissan-pow`, H200 per episode.

Arms differ only in memory/cache: **stock** 18/6; **budget** `--budget-memory` (K10 native, bounded 24-frame pose-indexed
KV archive, experimental); **cache46** `--local-attn-size 46 --sink-size 6` (experimental). The judge saw a neutral model name.

Primary score: PlayWorld's released rubric verifier via Vertex (`gemini-3.1-pro-preview`, gcloud login,
`worldsync/gemini_vertex_judge.py`). Secondary: our Opus judge. **Not comparable with the paper's 1.82 / 2.12**
(different checkpoint, player, video length and judge model; only GC and OE tasks, so no four-dimension overall).

## Per-arm Gemini score (28 episodes per arm; bootstrap 95% CI over episodes)

| Arm | Gemini mean [95% CI] | GC | OE | gate non-fail | Opus mean |
| --- | --- | ---: | ---: | ---: | ---: |
| stock | 1.81 [1.35, 2.35] | 2.46 | 1.16 | 21/28 | 2.17 |
| budget | 2.13 [1.66, 2.67] | 2.81 | 1.45 | 25/28 | 2.33 |
| cache46 | 2.26 [1.76, 2.83] | 2.53 | 2.00 | 21/28 | 2.28 |

## Paired deltas vs stock (28 (case, seed) pairs; 14 GC, 14 OE)

| Arm | Slice | Mean delta [95% CI, pairs resampled] | Case-clustered 95% CI (7 or 14 cases) | W/T/L |
| --- | --- | --- | --- | --- |
| budget | all | +0.32 [-0.08, 0.75] | [-0.12, 0.85] | 9/14/5 |
| budget | GC | +0.36 [-0.39, 1.13] | [-0.50, 1.36] | 6/5/3 |
| budget | OE | +0.29 [-0.03, 0.71] | [0.00, 0.63] | 3/9/2 |
| cache46 | all | +0.46 [-0.22, 1.15] | [-0.35, 1.21] | 12/10/6 |
| cache46 | GC | +0.07 [-1.07, 1.21] | [-1.30, 1.31] | 7/2/5 |
| cache46 | OE | **+0.84 [0.20, 1.62]** | **[0.14, 1.63]** | 5/8/1 |
| Opus: budget / cache46 | all | +0.09 [-0.05, 0.23] / +0.05 [-0.08, 0.18] | | 13/8/5 / 8/11/7 |

Per-seed Gemini means (stock / budget / cache46): seed 42 = 1.89 / 2.29 / 1.94; seed 43 = 1.72 / 1.97 / 2.59.

## Reading

* The only interval clearing zero is **cache46 on OE (+0.84)**, driven mostly by seed 43 (seed 42 gave +0.36 with a CI
  including 0). It is one of 8 Gemini contrasts, none corrected for multiple comparisons, and the Opus judge sees nothing
  (+0.05). Treat it as a lead, not a result.
* **Budget memory** is positive in both seeds (+0.40, +0.25 by arm mean) and cuts gate failures (3/28 vs 7/28), but its
  CI includes 0 overall and on GC; on OE the clustered interval just touches 0.
* GC shows no reliable effect of either arm. 46/6 needs 115 GB and about 1.45x time; budget fits at about 90 GB.
* Gemini vs Opus (82 scored episodes with both): Pearson 0.64, Spearman 0.67; Gemini 0.17 lower on average; gate pass
  rate Gemini 79% vs Opus 94%. The Gemini-only gains may be judge-specific.
* Player and judge noise dominates: stock alone moved 1.89 to 1.72 between seeds; many episodes sit at the gate-fail floor of 1.0.

## Limits

Two seeds, 14 cases, two dimensions of four, one judge model with no repeat calls (Gemini sampling noise unmeasured),
experimental arms, a different player/video length from the paper. Seed 44 was deliberately not run (shared capacity).

## Files

`powered-table.md`, `powered-analysis.json` (pairs resampled), `case-cluster-ci.json`, `agreement-and-categories.txt`,
`episode-meta.json`, `opus-seed42.json`, `opus-seed43.json`, `gemini-episodes/` (84 verifier outputs).
Code: `scripts/modal_lingbot_powered.py`, `scripts/download_powered.py`, `scripts/build_powered_manifest.py`,
`scripts/analyze_powered.py`, `scripts/powered_cluster_ci.py`, `scripts/powered_extra_stats.py`.
Run: `MODAL_PROFILE=nissan-pow modal run scripts/modal_lingbot_powered.py --seed 42` (then 43), download, Gemini judge, analysis.
Gemini category means in `agreement-and-categories.txt` are conditional on the gate and have unequal n; do not compare them across arms.
