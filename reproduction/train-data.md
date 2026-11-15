# Revisit training data (for the LingBot LoRA fine-tune)

Purpose: clips where the camera leaves a view and later returns to it, so a fine-tune
can learn to recall the earlier view. Built on Modal; nothing is stored in Git.

## Source and licence

* **Sekai** (Lixsp11/Sekai; real-world YouTube walking/drone video with camera
  trajectories), via the public repackaging `newhorizon2005/sekai960-memory-20261104`:
  1012 validated clips, each 960 frames at 16 fps (60 s), with normalized intrinsics and
  per-frame extrinsics.
* **Licence: non-commercial, research purposes only**
  (<https://github.com/Lixsp11/sekai-codebase/blob/main/LICENSE>). Any model fine-tuned on
  this data inherits that restriction; LingBot-World2 itself is CC BY-NC-SA 4.0, so the
  two are compatible. The mirror is third-party; the original source of record is
  Sekai. YouTube content remains subject to its owners' rights.
* Rejected sources: DL3DV (gated, and the Modal `huggingface` secret holds an invalid
  token), SpatialVID (gated, and its clips are only 3 to 15 s, too short for 8 s gaps),
  RealEstate10K (short clips, needs YouTube downloads).

## Interface for the trainer

Volume `worldsync-train-data` (Modal workspace `nissan-pow`), mounted at `/data`:

```
/data/manifest.jsonl                 one row per revisit pair
/data/clips/<clip_id>/video.mp4      832x480, 16 fps, H.264, 960 frames, no audio
/data/clips/<clip_id>/poses.npy      [960,4,4] float32, OpenCV camera-to-world
/data/clips/<clip_id>/intrinsics.npy [960,4] float32, pixels [fx, fy, cx, cy] at 832x480
/data/clips/<clip_id>/revisit.json   pairs and conversion metadata
/data/selection.json                 selection summary
```

Manifest row fields: `clip_id, video, poses, intrinsics, width, height, fps,
num_frames_clip, start, end, keyframe_frame, revisit_frame, keyframe_time_s,
revisit_time_s, gap_s, overlap_score, min_overlap_between, source_group, license`.

* `[start, end)` is the training window in frames (`end - start = 4k + 1`, as the
  LingBot pipeline expects). It begins ~2 s before the keyframe and ends ~4 s after the
  revisit. Slice `poses.npy` and the video with the same indices.
* `keyframe_frame` is the view that must be recalled; `revisit_frame` is where the camera
  returns. Both are absolute frame indices within the clip (subtract `start` for window
  coordinates).
* Poses are in the LingBot convention: `poses.npy` is OpenCV c2w. The pipeline converts
  to framewise relative poses itself (`compute_relative_poses`), so no world re-origin
  is needed. Intrinsics match the 832x480 layout of `external/LingBot-World2/examples`.

## How pose and intrinsics conventions were checked

* Extrinsics: measured on 40 clips, forward motion expressed in the camera frame is
  +0.92 along local z under the camera-to-world reading and +0.50 under world-to-camera,
  so they are treated as OpenCV c2w (`scripts/modal_traindata_cam_probe.py`). Frame 0
  is `diag(1,-1,-1)`, not identity, which is harmless after relative-pose conversion.
* Intrinsics are normalized by image size (`fx/W`, `fy/H`, ...); they are converted with
  the exact resize and centre-crop applied to the video (cover-scale to height 480, crop
  to 832 wide).

## What counts as a revisit (and its limits)

`worldsync/revisit_pairs.py` uses poses only: the later view's camera must be within
6% of the trajectory length of the keyframe camera and within 45 degrees of its heading
(overlap proxy >= 0.55), gap 8 to 24 s, and the camera must have left the view in
between (proxy <= 0.12). This is a candidate generator. It never checks scene content,
so some pairs may not show the same surface (for example a moving crowd, or monocular
pose drift). Treat `overlap_score` as a weight, not a label. Poses are monocular
estimates (MegaSaM) with normalized scale, so absolute distances are approximate.

## Rebuild

```
MODAL_PROFILE=nissan-pow modal run scripts/modal_build_revisit_data.py --target-clips 300
```

## Built set (2026-11-14)

* 325 clips on the volume (3.98 GB at 832x480), 376 revisit pairs, selected by pose
  with relaxed thresholds (`--radius-frac 0.12 --high 0.4 --max-per-group 6`; the strict
  default produced only 151 pairs in 133 clips).
* A content check (tiny grayscale frame differences: the keyframe must differ clearly
  from some in-between frame and resemble the revisit frame) flags pairs in the manifest
  as `content_ok`. **117 pairs in 111 clips pass.** Use `content_ok == true` for training
  unless more data matters more than precision; the rest are pose-only candidates, often
  from a near-static camera where pose noise looks like motion. Thresholds are heuristics
  (`content_diff_max_between >= 0.12`, `content_diff_revisit <= 0.6 * that`); the raw
  values are in each row for re-filtering.
* A visual spot check of 6 high/medium/low scoring strict pairs showed the revisit frame
  in the same scene each time; the top pairs have nearly identical framing.
* Typical window: ~233 frames (about 14.5 s). Gaps are mostly 8 to 12 s.
* Summary tool: `python scripts/summarize_revisit_manifest.py manifest.jsonl`.

## Known limits

* Only ~34% of Sekai clips contain any revisit candidate and about a third of those pass
  the content check, so the clean set is small (about 117 windows). That is enough for a
  LoRA smoke test, not a large run. Larger sets need content-based mining (feature
  matching across all 1012 clips, other Sekai splits) or synthetic loops rendered from
  3D scenes.
* The 8 to 12 s gaps are short compared with the 25 s plus episodes in PlayWorld.
* Captions and text prompts are not included; the trainer needs a generic prompt or
  the Sekai captions from `Lixsp11/Sekai` (`train/*.csv`).
