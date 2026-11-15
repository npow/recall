# Keyframe-restart memory (training-free diagnostic, not a PlayWorld score)

> **Integration note (main).** Only the helper module `worldsync/lingbot_keyframe_restart.py` (bank, trigger, seam helpers) with its
> tests, the report scripts and these results are on `main`. The `--keyframe-restart` / `--restart-*` rollout flags and
> `scripts/modal_lingbot_keyframe.py` are **not wired into main's `lingbot_rollout.py`**: the restart re-enters
> `pipeline.generate` from a stored keyframe, which needs a larger change to the rollout's control flow than a flag merge.
> The numbers below were measured on the fork's own rollout; they cannot be re-run from main yet.


`--keyframe-restart` in `worldsync/lingbot_rollout.py` (bank, trigger and seam helpers in
`worldsync/lingbot_keyframe_restart.py`, 12 tests). Default off; native fixed controls only; mutually exclusive with
the other memory flags; not combinable with `--offload-text-encoder`.

**Idea.** Keep real generated keyframes together with the commanded camera pose at the instant they were shown (the
start image is keyframe 0). When the commanded pose re-enters a stored keyframe's neighbourhood (position and
heading thresholds) after a gap longer than the KV window (5 chunks vs the 4.5-chunk window), raise out of the
denoiser at a chunk boundary and start a new `pipeline.generate` from the stored keyframe image: fresh KV caches,
fresh streaming-VAE state, native pose track from the current commanded pose, different noise seed per segment.
Optional: crossfade of the first N frames (`--restart-crossfade-frames`), a drift gate (`--restart-drift-threshold`),
an excursion rule (`--restart-min-excursion-degrees/-position`: the camera must have left the keyframe pose since it
was shown, so a long hold on one view is not a "revisit") and a cooldown (`--restart-cooldown-chunks`).

**Cheating risk, stated up front.** A restart makes the revisit match the stored view by construction. DINO against
the start view therefore measures the mechanism, not the world model. What matters is (1) whether the trigger fires
at the right time, (2) the visible cut, and (3) what is lost: every restart discards the model's own state, so
anything that moved or changed since the keyframe snaps back. All four test scenes here (GC007 facade, GC001
interior) are static, so the cost of (3) is **not measured** and cannot be seen in these runs.

Setup: GC007 (distant facade) and GC001 (indoor, near-field), seeds 42 and 43, 68 deg/s turns, one Modal H200 per job,
stock 18/6 cache, fixed actions, DINOv2-small CLS cosine of the final frame against frame 12 and frame 0
(`scripts/keyframe_restart_report.py`). No judged score, no adaptive player.

## Results

Scheme B (the default; the conditioning frame sits one latent step ahead so the segment stays on the commanded
timeline) with the excursion rule (30 degrees / 2 units) and a 1-chunk cooldown. DINO final vs frame 12 (f12):

| Path | Scene | stock s42 / s43 | budget K10 native (s42) | 46/6 (s42 / s43) | **keyframe restart** (s42 / s43) | restarts at chunk |
| --- | --- | --- | --- | --- | --- | --- |
| turn_return | GC007 | 0.581 / 0.700 | 0.773 | 0.818 / 0.818 | **0.975** / 0.989 (scheme A; scheme B seed 42: 0.974) | 7 |
| turn_hold_return | GC007 | 0.304 / 0.404 | 0.768 | 0.899 / 0.878 | **0.989 / 0.978** | 11, 12, 13 |
| turn_hold_return | GC001 | 0.134 / 0.386 | n/a | 0.679 / n/a | **0.953 / 0.975** | 11, 12, 13 |
| turn_return | GC001 | 0.491 / 0.473 | n/a | 0.744 / n/a | **0.970 / 0.981** (scheme A) | 7 |
| translate_return | GC001 | 0.755 | n/a | n/a | **0.984** (scheme B) / 0.927 (scheme A) | 8, 9 |

Baselines for stock, budget memory and 46/6 are from `reproduction/revisit/` and `reproduction/budgetmem/`
(GC007 stock seed 43 and GC001 stock seed 43 and the translate path stock were rerun here). Peak GPU memory is the
stock 80.2 to 80.3 GB and job time is within noise of stock (115 to 190 s): the method has no memory or time cost.
Full per-run numbers: `report.json`.

Dated schemes in the table: the first runs used scheme A (conditioning at the last emitted frame), which leaves the
camera one latent frame (up to 17 degrees of a turn) behind the commanded pose after each restart. Scheme A
turn_return numbers are kept because the path restarts only once and ends on a WAIT chunk, where the lag does not matter.

### What the runs show, including the failures I hit on the way

1. **The simple trigger works only when the camera returns to its pose while moving.** On `turn_return` it fires
   exactly once, at the final WAIT chunk (chunk 7), and the final frame is the start view (DINO 0.97 to 0.99).
2. **v1 failed on `turn_hold_return`.** With no excursion rule the trigger fired at chunk 9, in the middle of the
   hold at the far heading (a keyframe from the same continuous view), which started the cooldown and blocked the
   real return at chunk 13. Final DINO 0.27 to 0.57: no better than stock. (`restart-*-turn_hold_return-*` rows.)
3. **v2 and v3 (excursion rule) still failed the return** for a different reason: the first valid revisit is the
   outbound view at the same heading during the sweep back (chunk 11), and scheme A then lags the commanded timeline
   by one latent frame (17 degrees), so at chunk 13 the camera was 17 degrees from the start and the 15-degree
   heading threshold rejected it. Final DINO 0.49 to 0.64 (better than stock's 0.13 to 0.40, not a return).
4. **Scheme B (aligned conditioning) fixed it:** restarts at 11, 12 and 13 (every chunk of the sweep back), final
   DINO 0.95 to 0.99 on all four hold runs.

### Visual continuity and seams

- Hold path, GC007 seed 42 (`strip-B-GC007-hold.png`: frame 12 | before and after each cut | final): the model's own
  views during the sweep back have drifted (garbled sign text, wrong storefront); each restart replaces the view
  with the stored outbound keyframe at that heading. The three cuts are abrupt jumps of viewpoint and content. The
  final frame is the true start facade.
- Seam size: frame-to-frame mean absolute difference across the cut vs the median difference of the same video:
  1.7 to 2.7x on `turn_return` (where the camera is turning anyway), 22 to 49x on the hold path (the typical difference
  is near zero while the camera is held, so any cut stands out), 3.6 to 5.8x on translate. DINO between the frame
  before and after a cut is 0.33 to 0.93.
- Crossfade works as a seam fix: with `--restart-crossfade-frames 8` on GC007 `turn_return` (scheme A) the seam
  drops from 2.7x to 0.3x the typical difference and the final DINO is unchanged (0.974). It was tested on one run
  only and only on the turn path; it turns the cut into an 8-frame dissolve between two different views.
- Translate (`strip-B-GC001-translate.png`): the cut at chunk 8 removes an object the model had invented (a dark
  side table) and returns to the room as stored; the cut is small and the final frame matches the start.

### What is lost, and what is not shown

- Dynamic state. Static scenes only; moving people, water, lights and objects are reset to the keyframe at each
  restart. This is the central cost of the method and is not measured here. The `turn_return_onward` test (restart
  at chunks 5 and 8 with `--restart-min-gap-chunks 2`, then turning right again) ended at the same DINO as stock
  (0.509 vs 0.508): restarting at one instant does not help the content generated afterwards.
- Every restart consumes one latent step: three native frames are missing from the video (212 frames on the hold
  path with three restarts, vs 221 for stock).
- Pose mismatch. The keyframe is accepted when the pose is within 1 unit and 15 degrees of the keyframe's pose, but is
  then treated as if it were exactly at the current pose. A larger threshold would make the camera jump.
- Revisits that happen while the camera is moving through the pose are handled by the same chunk-boundary check
  and are therefore quantized to chunk starts.
- Not tested: adaptive player episodes (the trigger needs the commanded pose, which the player's CameraPlan does
  provide, but `--keyframe-restart` is wired for native fixed controls only), other scenes, more seeds, a judged
  score, a drift gate other than "always" and crossfade on the hold path.

## Reproduce

```
MODAL_PROFILE=nissan-pow modal run scripts/modal_lingbot_keyframe.py --lanes 2 --skip 1   # scheme A matrix
MODAL_PROFILE=nissan-pow modal run scripts/modal_lingbot_keyframe.py --lanes 2 --b         # scheme B hold/turn/translate
python scripts/keyframe_restart_report.py --runs label=<run dir> ... --output out/
python scripts/restart_seam_strip.py --run <run dir> --output strip.png
```

Raw run directories are in the Modal results volume `worldsync-lingbot-results` under `kf-*`.
