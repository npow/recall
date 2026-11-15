# Text director: development experiment (2026-11-15, option B)

Question: can LLM-written world-model text lift the PlayWorld floor (frozen scenes, no evolution) enough to
move the overall from 1.85 toward 2.12? Short answer: **no reliable gain; at best about +0.1 to +0.2 overall,
inside the noise.** Full numbers: `director-dev/table.json`, `director-dev/table.txt`.

## Setup

* 20 development cases, never chosen by outcome (`scripts/director_dev_cases.py`): 6 out-of-sight, 5 IF, 3 GC from
  the development split by salted hash, plus the six declared insight development cases.
* Every episode is a **fixed-action replay** of the stock full-run trajectory for that case (same seed 42, same
  commanded actions, same frame budget), so only the text or the inference knob differs. No player calls.
* Generated on Lambda H100 PCIe. Control = `replay` (stock text) on the same hardware, because outputs are
  bit-exact only within one GPU model.
* Judge: released PlayWorld rubric/gate/aggregation through Vertex `gemini-3.1-pro-preview`, **three independent
  calls per video**, scores are the mean of three. Paired delta = variant minus replay per case; CI = percentile
  bootstrap over cases.
* Arms: `lstatic` = `--director static`, texts by Claude Haiku 4.5 through `worldsync.lingbot_director.Director`
  (clean-room: caption, objective, reference plan only); `lsched` = `--director unseen` (texts refreshed every 4
  chunks and at action changes, view-aware); `nag` = `--nag` (scale 8, tau 2.5, alpha 0.25, negative prompt
  "static, still, frozen, ..."); `sink2` = `--sink-size 2`.

## Result (stages 0-3 of the job list, scored by 04:41 PST)

| Arm | Dimension | n | variant | replay control | paired delta [95% CI] | W/T/L |
|---|---|---:|---:|---:|---|---|
| lstatic | IF | 5 | 1.86 | 1.84 | +0.03 [-0.89, +0.99] | 1/2/2 |
| lstatic | out-of-sight | 6 | 1.33 | 1.00 | +0.33 [+0.03, +0.67] | 3/3/0 |
| lstatic | insight | 6 | 1.99 | 1.56 | +0.43 [-0.08, +0.98] | 3/1/2 |
| lstatic | GC (no-harm check) | 3 | 1.89 | 2.33 | -0.44 [-1.33, +1.33] | 1/0/2 |
| lsched | out-of-sight | 4 | 1.08 | 1.00 | +0.08 [0.00, +0.25] | 1/3/0 |
| nag | IF | 5 | 1.60 | 1.84 | -0.24 [-0.64, 0.00] | 0/3/2 |
| nag | out-of-sight | 6 | 1.04 | 1.00 | +0.04 [0.00, +0.13] | 1/5/0 |
| sink2 | out-of-sight | 6 | 1.06 | 1.00 | +0.06 [0.00, +0.17] | 1/5/0 |

Hardware/judge noise check, same config on two GPU models (Lambda replay vs the Modal H200 stock video):
IF 1.84 vs 1.53, out-of-sight 1.00 vs 1.03, insight 1.56 vs 1.47, GC 2.33 vs 3.51.
Within-video judge SD over three calls: 0.35 (80 videos). GC001 on identical text got 5, 1, 5 and 1, 1, 5.

## Reading

* **IF: no effect.** Stating the subject's commanded motion in the text did not raise IF (+0.03).
* **Out-of-sight: +0.33**, the only interval that clears zero, but it comes from two cases (OE067, OE068) where
  ONE of three judge calls gave 3.5-4.0 and the other two gave 1.0. Four of six stay at the floor on all calls.
* **Insight: +0.43, interval includes zero**; driven by OE074 (1.33 to 2.93) and OE008 (2.8 to 3.67, one call 1.0).
* **GC: -0.44 on n=3**, all from single judge calls flipping between 1 and 5; not evidence of harm, but no evidence
  of safety either.
* The scheduled (per-chunk) director, NAG and sink-size 2 show nothing on out-of-sight; NAG lowers IF.
* If the lstatic deltas transferred exactly (director on IF, insight and out-of-sight, stock on GC) the overall
  would move by about (0.03 + 0.43 + 0.33 + 0) / 4 = **+0.20, to about 2.05**; with the director on GC too and the GC
  dev delta taken at face value, +0.09. Both are below the +0.27 needed for 2.12, and the dev intervals are wide.

## Not done / limits

* Insight `lsched`, `sink2`, `nag` and the in-session preset texts (stages 6-12) were not scored by the cut-off
  (Lambda throughput was far below plan); KV recache at a prompt switch was not implemented.
* The in-session preset texts (`director-dev-texts.json`) were only run on Modal for 25 short episodes before Modal
  was shut down by user directive; they were not scored into this table.
* One seed, 3-6 cases per dimension, replayed trajectories (the adaptive player is not in the loop).
* NAG was run for the first time here; scale/tau/alpha were not tuned.

## If one configuration must be frozen

`--director static --director-text-file reproduction/director-static-texts-all.json` on IF, insight and
out-of-sight; GC unchanged (stock text). The file holds clean-room Haiku texts for all 171 cases (0 LLM errors;
the 20 development cases reuse the texts tested here). It needs no LLM call at run time.
