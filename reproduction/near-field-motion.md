# Is LingBot-World2's camera control realized? Near-field retest

Question: the realized-motion check on GC007 found rotation realized at about 0.9 of commanded but forward translation barely
moving (about 1.04x scale over 4 W chunks). Is that a control failure, or a property of that distant-facade scene, and does it
explain the instruction-gate failures (GC005, GC007, GC027, GC017, GC015, GC024)?

**Answer: mostly scene-dependent.** Forward, backward and turning are realized in 12 of 13 scenes. GC007 (a far facade) is
the outlier, and it is the only gate-failure scene where translation is actually weak. For the other five gate failures the
commanded camera motion happens, so control realization is not what fails them.

Stock checkpoint (`lingbot-world-v2-14b-causal-fast`), stock 18/6 window, seed 42, no memory method, no quantization, fixed
native control paths, no player and no judge. Monocular odometry on generated video: scale-free and noisy, good for comparing
actions and scenes, not metric motion. Not a PlayWorld score.

## Setup

* 56 jobs on the existing persistent runner (`worldsync-lingbot-runner`, Modal workspace `nissan-pow`, H100-class settings:
  expandable-segments allocator, text encoder offloaded after encoding, at most 2 containers). All 56 completed, weights
  checksum unchanged after every job, peak allocated 68.6 to 72.0 GB, mean 52 s per job, 1,672 s wall for the whole matrix
  (one-shot jobs would have each paid the 100 to 150 s model load). Job list: `near-field-jobs.jsonl`. Raw per-chunk numbers:
  `near-field-motion/measurements.json`; runner results: `near-field-motion/runner-results.json`.
* The two new Modal workspaces (`krasnaya-zvesda`, `seraphina-ink-88`) cannot launch GPU functions: Modal answers "Please add a
  payment method to use H100 GPU functions" (also H200 and L40S). I did not use them for this test.
* 13 scenes. Gate-failure scenes: GC017, GC015, GC024, GC007, GC005, GC027. Near-field scenes picked from the start images
  (interiors and narrow streets): GC011 entryway, GC012 open-plan interior, GC014 hospital ward, GC023 living room, GC025
  doorway into a bedroom, GC028 corridor. Controls: GC001 interior and GC007 far facade (the earlier finding).
* Paths (one action per chunk, the first chunk is WAIT): `fwd8` W x8, `bwd8` S x8, `left8` A x8, `right8` D x8, `turn180` RIGHT
  x3 (commanded 180 degrees), `round_trip` (W x4, RIGHT x3, W x4, RIGHT x3). Forward and backward and turn ran on all 13 scenes,
  strafes on 6, the round trip on 5.
* Measurements per chunk (`scripts/measure_near_field.py`): similarity-transform scale between the chunk's first and last
  frame from SIFT matches (above 1 the scene expands, so the camera approached; below 1 it retreated); the horizontal scene
  shift `tx` in px at 832 px width (A strafe moves the scene right, +); yaw from the essential matrix
  (`worldsync/motion_feedback.py`, + = right). "Cumulative" is the product (scale) or sum (tx, yaw) over the action's chunks.

## Results (cumulative over the action's chunks)

| scene | W x8 scale | S x8 scale | A x8 tx px | D x8 tx px | RIGHT x3 yaw (180 cmd) | round trip: W scale over its 8 W chunks | gate failure? |
|---|---|---|---|---|---|---|---|
| GC001 | 1.87 | 0.56 | 113 | -34 | 185 | 1.27 | control |
| GC005 | 1.78 | 0.39 |  |  | 163 |  | yes |
| GC007 | **1.07** | **0.83** | 53 | -1 | 178 | 1.18 | yes, and control |
| GC011 | 2.08 | 0.30 | 147 | -77 | 178 |  |  |
| GC012 | 1.42 | 0.59 |  |  | 169 |  |  |
| GC014 | 2.12 | 0.40 |  |  | 172 |  |  |
| GC015 | 3.55 | 0.27 | 201 | -20 | 170 | 2.64 | yes |
| GC017 | 1.71 | 0.48 | 154 | -77 | 179 | 2.95 | yes |
| GC023 | 2.66 | 0.50 |  |  | 170 |  |  |
| GC024 | 1.62 | 0.62 | 127 | -196 | 184 | 1.34 | yes |
| GC025 | 4.79 | 0.49 |  |  | 173 |  |  |
| GC027 | 1.35 | 0.68 |  |  | 195 |  | yes |
| GC028 | 1.18 | 0.30 |  |  | 180 |  |  |

Summary over the 13 scenes:

* **Forward (W):** every scene expands (median 1.78x, range 1.07 to 4.79). 11 of 13 reach at least 1.3x. Weak: GC007 (1.07)
  and GC028 (1.18, but its backward motion is a strong 0.30).
* **Backward (S works):** every scene contracts (median 0.49, range 0.27 to 0.83). 12 of 13 are at or below 0.7. Weak: GC007
  (0.83). So the S action is not broken: GC024 retreats (0.62) and GC007's S is the weakest in the matrix only because the
  facade is far.
* **Turn in place:** realized yaw is 163 to 195 degrees for 180 commanded (mean 176.6, 98 percent), in every scene, with no
  dependence on scene distance.
* **Strafe:** A moves the scene right (+53 to +201 px) and D moves it left (-1 to -196 px) in all 6 scenes, so the directions are
  right, but D is weaker than A in 5 of 6 (mean +132 against -67; GC007 D is about 0).
* **Near against far:** the 9 interior and near scenes (GC001, GC011, GC012, GC014, GC015, GC017, GC023, GC025, GC028) have
  median W scale 2.08. The far or mid-range scenes (GC005 statue plaza 1.78, GC024 street 1.62, GC027 playground 1.35, GC007
  facade 1.07) are lower, and the farthest, GC007, barely moves.
* **Hardware cross-check:** the same GC017 jobs on the local Thor GPU (different GPU and PyTorch) gave W 1.66 and S 0.43 against
  1.71 and 0.48 on Modal (`near-field-motion/measurements-thor-crosscheck.json`), so the numbers are not a Modal artifact.

Images: `near-field-motion/strip-forward.jpg` (frames at 0, 1/3, 2/3 and the end of W x8 for GC007, GC027, GC028, GC025, GC011)
and `strip-backward-turn.jpg` (S x8 for GC007, GC024, GC001; turn in place for GC017 and GC027). GC007 and GC027 show the facade
only growing slightly; GC028, GC025 and GC011 walk into the room.

## What this says about the gate failures

Task text and the human reference plans are from `external/PlayWorld/data/gc/data.json`.

| task | what it needs | realized here | so the failure is probably... |
|---|---|---|---|
| GC007 walk to the Primemall entrance, return, face the start (plan W 7.2 s, S 7.2 s) | large approach to a far facade | **weak**: W 1.07, S 0.83 over 8 chunks | control realization (translation to a far target is not realized): the one real case |
| GC017 walk forward 10 steps, then turn back | forward and a turn | W 1.71, turn 179 degrees | not control; player or judge (premature END, or "turn back" ambiguity) |
| GC015 walk forward 5 steps, back 5, turn to look at the start | forward, back, turns | W 3.55, S 0.27, turn 170 | not control; player or judge |
| GC027 rotate 360 degrees in place, return to the original angle | pure rotation | turn 195 for 180 commanded | not control; player or judge (the rotation is realized, so check how the player and the rubric treat a 360-degree return) |
| GC024 the person with the backpack walks past the hot dog stand, then walks backward in place | subject motion on a busy street | W 1.62, S 0.62 | not camera control: the task is about a dynamic subject, which camera poses cannot direct |
| GC005 orbit a statue, return (plan uses W, D, A, S) | strafes plus translation | W 1.78, S 0.39, turn 163; strafe not run on GC005 itself | probably D-strafe, which is the weakest action elsewhere (about half of A); untested on this scene |

So control realization is a real but narrow bottleneck: far-field translation (GC007) and, possibly, the D strafe.
It does not account for the other four or five gate failures. The earlier recommendation to prioritize control realization should
be downgraded: the closed-loop feedback arm helped GC007 and GC005 episodes, which is consistent with this (those are the weak
cases), but its noisy paired gain should not be read as a general fix.

## Failures worth keeping

* **Forward runs into obstacles and degrades.** 8 W chunks in the near-field scenes walk the camera into walls: the last frames of
  GC028 (door frame), GC025 (wallpaper) and GC011 (a smeared close-up texture) are no longer coherent scenes. Plans of 2 to 4 s do
  not reach this, but the long forward moves in the longer reference plans (GC007 7.2 s) would, in near scenes.
* **Content behind the camera is invented.** On S the model synthesizes what was behind the start view, so a backward path does not
  retrace what a forward path saw; GC001, GC015 and GC017 backward frames show new rooms and street.
* **D strafe is weaker than A strafe** in 5 of 6 scenes (GC007 D is about 0 px).
* **Magnitude is unknown.** Scale ratios say a motion happened, not that it matches a human step: "10 steps", "7.2 s forward" have no
  calibrated meaning here, and GC015's 3.55x over 8 chunks is far more than GC007's 1.07x for the same commanded motion.

## Caveats

* One seed (42) per cell, one scene image per scene. Seed noise on the scale measurement was not estimated.
* Monocular, scale-free odometry on generated video; the scale is a median over matched features, dominated by whatever
  structure SIFT finds, so low scale partly reflects far content, which is the finding but also a measurement limit.
* No judged score and no player in these runs. "Probably" in the gate table is inference from commanded-motion realization plus the
  task text, not a test of those episodes; GC005 strafe and the player behaviour on GC017/GC015/GC027 were not run here.
* `krasnaya-zvesda` and `seraphina-ink-88` were not used for compute (no payment method). The seeding script
  `scripts/modal_seed_weights.py` and an H100-only runner variant `scripts/modal_lingbot_runner_h100.py` exist for them, but
  are unused and untested past the payment error. The weights seeded into `krasnaya-zvesda` came from the Hugging Face repo head
  and were not hash-checked against `reproduction/lingbot-weights.json`; do not use them for scored work before verifying.

## Reproduce

```bash
python3 scripts/near_field_jobs.py reproduction/near-field-jobs.jsonl          # 56 jobs
MODAL_PROFILE=nissan-pow <modal python> scripts/lingbot_runner_client.py reproduction/near-field-jobs.jsonl \
    --gpu H100 --containers 2 --out outputs/nf-runner-results.json              # needs the deployed runner app
MODAL_PROFILE=nissan-pow python3 scripts/fetch_nf.py outputs/nf
<python with opencv> scripts/measure_near_field.py outputs/nf/nf-* --output outputs/nf-measure.json
python3 scripts/nf_table.py outputs/nf-measure.json --markdown
```
