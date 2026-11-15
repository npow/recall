# Closed-loop motion feedback on the gate-failing GC cases

Local Opus judge, seeds 42 and 43, `--end-guard` on, stock 18/6 cache, one H200 per episode.
**Not comparable with the paper (1.82 / 2.12).** Baseline = the earlier guarded stock rows
(`guarded-results.json`), a separate player and generation draw, so differences are noisy
(identical settings swing up to about 2 points between draws). Raw rows: `feedback-results.json`
(produced by `scripts/aggregate_feedback.py`).

Arms (both experimental deviations, see `reproduction/lingbot.md`):
* `fb`: the player sees measured visual-odometry yaw and image motion per chunk.
* `fbcomp`: `fb` plus turns commanded at 68 deg/s (player-facing odometry kept at 60 deg/s).

| Arm | Scored | Mean score | Gate fail / partial / pass |
| --- | --- | --- | --- |
| stock baseline (6 cases x 2 seeds) | 12 | 2.50 | 3 / 8 / 1 |
| fb | 9 of 12 | 2.92 | 0 / 9 / 0 |
| fbcomp | 9 of 12 | 2.71 | 1 / 8 / 0 |

Paired against the baseline where both were scored: `fb` mean +0.37 (4 wins, 1 tie, 4 losses over 9
pairs); `fbcomp` mean -0.29 (4 wins, 5 losses over 9 pairs).

| Case, seed | Baseline | fb | fbcomp |
| --- | --- | --- | --- |
| GC005 s42 | 1.0 fail | generation failed | generation failed |
| GC005 s43 | 1.0 fail | 2.4 partial | generation failed |
| GC007 s42 | 1.0 fail | 3.2 partial | generation failed |
| GC007 s43 | 1.8 partial | 3.2 partial | 1.0 fail |
| GC027 s42 | 2.6 partial | generation failed | 2.8 partial |
| GC027 s43 | 3.6 pass | 2.6 partial | 2.4 partial |
| GC017 s42 | 3.8 partial | 3.3 partial | 4.0 partial |
| GC017 s43 | 3.4 partial | 3.4 partial | 3.8 partial |
| GC015 s42 | 3.4 partial | generation failed | 2.4 partial |
| GC015 s43 | 2.1 partial | 2.4 partial | 1.4 partial |
| GC024 s42 | 3.1 partial | 2.8 partial | 2.8 partial |
| GC024 s43 | 3.2 partial | 3.0 partial | 3.8 partial |

## Reading

* **The three baseline gate failures** (GC005 s42 and s43, GC007 s42): `fb` converted two to partial
  (GC005 s43 1.0 to 2.4, GC007 s42 1.0 to 3.2; the third generation failed). That is the direction the
  score-driver analysis predicted, but it is two episodes.
* **`fb` also lost the only baseline pass** (GC027 s43 3.6 to 2.6) and was lower on 4 of 9 pairs, so
  the paired gain is not established: n is 9, a single episode swings the mean by about 0.3 to 0.5,
  and baseline-to-baseline draw noise is larger than the +0.37.
* **`fbcomp` did not help** on this sample (mean paired -0.29; GC007 s43 turned into a gate fail).
  68 deg/s was calibrated from one video; its effect is not separable from noise here.
* **Reliability problem:** 6 of 24 episodes failed to generate, all with `PlayerResponseError: invalid
  player action` (the Haiku player returned a non-END operation with action NONE or none). The baseline
  had 3 generation failures in 60. The added prompt hint ("Do not END while measured motion shows ...")
  may be raising this, but I did not test that. A retry on invalid actions would remove most of the
  loss and was not added in the time available, so the arms are unbalanced (9 scored of 12 each).
* **Did the player act on the measurement?** Not analysed per episode (per-chunk measurements are in
  each episode's `run.json` `motion_feedback`, in Modal volume `worldsync-lingbot-results`, `cohort-guard-fb-*`
  and `cohort-guard-fbcomp-*`). So the mechanism (extended movements after low image motion) is unverified.
* **Estimator limit:** yaw is validated against one stock video (358 of 360 deg over two 180-degree
  turns). The image-motion value is a coarse proxy with heuristic thresholds from that same video, and
  monocular odometry has no absolute scale, so it cannot confirm that a distant landmark was approached.

Verdict: closed-loop yaw feedback is plausible and is the only arm that moved gate failures, but this
sample (9 scored episodes per arm, noisy baseline, 25% generation failures) does not establish an
improvement. A decisive test needs the invalid-action retry, a same-code stock rerun as the paired
baseline, and more seeds.
