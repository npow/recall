# Can text prompts make LingBot carry out commanded camera motion?

Short answer: **no, not in these runs.** No prompt variant raised forward approach, and a
post-normalisation translation gain of 2 changed trajectories unpredictably. These are fixed-path
diagnostics (not PlayWorld scores), one seed (42), stock 18/6 cache, local H100 80 GB runs.

## Mechanism checked (external/LingBot-World2, unmodified)

* The prompt is encoded once per `generate()`; the cross-attention K/V cache is built at the first
  denoiser call (`cross_attn_first_call`, `_cross_attn_initialized`). Passing `cross_attn_first_call=True`
  on the first call of a chunk rebuilds that cache from a new `context`, so a per-chunk prompt is possible.
* Upstream divides per-frame translations by the sequence maximum (`compute_relative_poses`,
  `normalize_trans=True`), so uniformly rescaling poses cancels. The only way to change the translation
  magnitude the model sees is a gain after normalisation (`--translation-gain`).

## Flags (default off; fixed-path diagnostics only)

`--motion-prompt {off,static,perchunk}`, `--translation-gain g` in `worldsync/lingbot_prompt_motion.py`,
wired into `worldsync/lingbot_rollout.py`. They require `--controlled-ablation --native-fixed-controls
--fixed-actions`, and the native-vs-online camera audit still passed with a gain (both paths patched).
Driver: `scripts/modal_lingbot_prompt_motion.py`. Measurement: `scripts/measure_prompt_motion.py`.
H100 fit without quantization: text encoder offloaded after encoding plus
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` gave 71.9-72.3 GB peak allocated (stock without these was 80.2 GB).

## Results (`measurements.json`; images `GC001-forward8-grid.jpg`, `GC007-forward8-grid.jpg`)

Approach scale = product over forward chunks of the similarity-transform scale between a chunk's first and
last frame (>1 means the scene expands, i.e. the camera moved forward). Monocular and scale-free: compare
variants only. Rows in the images: stock, static, per-chunk, per-chunk+gain2, stock+gain2 (left first frame, right last).

| scene | path | prompt | gain | approach scale | fwd flow px/s | yaw per turn chunk |
| --- | --- | --- | --- | --- | --- | --- |
| GC001 | forward8 | off | 1 | 1.80 | 9.2 | n/a |
| GC001 | forward8 | static | 1 | 1.61 | 6.9 | n/a |
| GC001 | forward8 | perchunk | 1 | 1.59 | 7.5 | n/a |
| GC001 | forward8 | perchunk | 2 | 1.76 | 9.4 | n/a |
| GC001 | forward8 | off | 2 | 1.89 (7 valid chunks) | 11.6 | n/a |
| GC007 | forward8 | off | 1 | 1.07 | 1.4 | n/a |
| GC007 | forward8 | static | 1 | 1.02 | 0.4 | n/a |
| GC007 | forward8 | perchunk | 1 | 1.04 | 0.5 | n/a |
| GC007 | forward8 | perchunk | 2 | 1.00 | 0.2 | n/a |
| GC007 | forward8 | off | 2 | 1.08 | 2.4 | n/a |
| GC001 | round_trip | off / perchunk / perchunk+g2 | 1 / 1 / 2 | 1.21 / 1.23 / 1.31 | 19.7 / 12.7 / 8.8 | 59.5 / 57.9 / 54.1 |
| GC007 | round_trip | off / perchunk / perchunk+g2 | 1 / 1 / 2 | 1.19 / 1.14 / 1.04 | 3.2 / 6.2 / 18.9 | 59.6 / 56.0 / 49.5 |

## What this shows

1. **Prompts do not help.** Static and per-chunk motion sentences give equal or lower approach than the stock prompt
   (GC001 1.80 -> 1.59-1.61; GC007 1.07 -> 1.02-1.04). The frames look the same (see images).
2. **Translation gain 2 is not a fix.** Approach moves within noise (GC001 1.80 -> 1.89 stock+gain, 1.59 -> 1.76 per-chunk+gain),
   but the end views differ (the camera drifts toward other parts of the room) and yaw per turn chunk drops
   (59.5 -> 54.1 on GC001, 59.6 -> 49.5 on GC007 in the round trip), so gain trades rotation fidelity for nothing reliable.
3. **The earlier "forward translation barely moves" claim is scene dependent.** In the near-field interior (GC001) stock forward8
   expands the scene by about 1.8x over 8 chunks and the video visibly walks into the room. In GC007 the facade is far away, so
   the same motion gives only about 1.07x. Measured turn yaw is about 59.5 deg per 60-deg chunk with this estimator on stock paths.
   So a far-landmark scene makes translation look absent; this run does not show the model ignores forward commands.

## Caveats

One seed per cell, two scenes, monocular odometry without absolute scale, and the scale estimate fails on some chunks
(valid forward chunk counts differ, e.g. GC007 round trip stock 7 vs per-chunk+gain2 4), so round-trip rows are not strictly comparable.
Per-chunk switching fired at action changes only (forward8: one switch, round trip: five; continuity looked normal in the frames,
but I did not run a separate seam test). No judged score and no quantization.
