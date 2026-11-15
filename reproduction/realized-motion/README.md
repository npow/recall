# Realized camera motion vs commanded path (round_trip, GC007, seed 42)

**Verdict: the camera does not return.** Heading roughly closes (about 20 degrees short).
Translation is barely realized, so position cannot close, and the end view is not the start
view. This is one scene and one seed, measured with monocular SIFT visual odometry on the
generated video; read it as a diagnostic, not a benchmark result.

Method: `scripts/realized_motion.py` matches SIFT features between frames 4 apart, fits an
essential matrix with the native intrinsics and reads yaw from the recovered rotation.
`scripts/compare_realized_motion.py` sums yaw per commanded turn block (steps with at
least 30 inliers only) and measures approach scale (similarity-transform scale between
frames 16 apart) per forward block.

## Stock run (commanded 180 degrees per turn)

| Block | Commanded | Realized | Notes |
| --- | --- | --- | --- |
| Forward, frames 12-76 | 4 unit steps | scale x1.044 total | facade expands about 4%, so very little approach |
| Turn 1, frames 76-124 | 180 deg | **165 deg** (12/12 reliable steps) | 92% of commanded; per-step 10-17 deg vs 15 |
| Forward, frames 124-188 | 4 unit steps | scale x1.054 total | consistent parallax, but in newly generated content |
| Turn 2, frames 188-236 | 180 deg | 44 deg measured; 174 extrapolated | only 3 of 12 steps reliable, so unmeasurable |

Net heading error is about 360 - (165 + about 174) = roughly 21 degrees. The second turn
is the weak point: after the first turn the scene is regenerated content with few matchable
features, so we cannot confirm it. Plot: `cumulative_yaw_stock.png`.

## Calibration variants (one H200 each)

| Variant | Turn 1 commanded / realized | Forward approach scale | Verdict |
| --- | --- | --- | --- |
| stock (60 deg/s, trans x1) | 180 / 165 | 1.044 | baseline |
| 65.5 deg/s, trans x1 | 196.5 / 174 | 1.038 | realized/commanded 0.89, so under-rotates about 10% across rates |
| 60 deg/s, trans x3 | 180 / 117 | 1.029 | worse: out-of-distribution translation hurts turning and does not add approach |
| 65.5 deg/s, trans x3 | 196.5 / 142 | 1.027 | worse |

* Rotation is monotone in the commanded rate: realized is about 0.89-0.92 of commanded.
  About 68 deg/s would give about 180 realized. That is a small, optional correction.
* Scaling translation after the pipeline's unit-max normalization (a diagnostic patch,
  `WORLDSYNC_TRANSLATION_SCALE`) did **not** increase approach and degraded turns. There
  is no calibration that closes the position loop. Do not keep it.
* Variants only changed control; none was judged for end-view recovery.

## What it means

The failures are not only memory. Rotation is faithful enough (about 90%), but forward
translation barely moves a distant facade, so a "forward 4 s, turn, forward 4 s, turn"
path does not trace a closed loop in the generated world. Memory methods cannot be
credited or blamed until translation is realized, for example with near-field landmarks
and measurable parallax. Limits: monocular VO has no scale, 9 of the 12 steps in the
second stock turn are unreliable, and nothing here is a PlayWorld score.
