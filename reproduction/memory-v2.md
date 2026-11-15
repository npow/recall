# Memory v2: store the frame, recompute the rest

Follow-up to the full-run null of budget memory ([audit](full-run-memory-audit.md) on `main`). Exploratory; one seed;
not a PlayWorld leaderboard score. Code: `worldsync/lingbot_memv2.py`, `worldsync/lingbot_memv2_recompute.py`,
flag `--memory-v2 {disk,recompute}` (default off). Config frozen before step 2: [memory-v2-config.json](memory-v2-config.json).

**Result in one line.** On scripted revisit paths v2 matches or beats the old method and removes its harm; on real
adaptive revisit episodes it does not improve the judged score or the judge-independent return similarity.

## What changed against budget memory

| Flaw found by the audit | v2 |
|---|---|
| 24-frame archive acts as a ~6 s FIFO on moving paths and keeps the newer, drifted copy | Nothing is evicted. `recompute`: keep each chunk's clean latent, image condition and camera embedding (about 6 MB per chunk) and recompute K/V of retrieved frames with a nested t=0 pass. `disk`: exact per-layer K/V of every frame on local disk (1.28 GB per latent frame, deleted at episode end) |
| Retrieval trusts commanded poses; translation is not visually undone | Pose stays the index (one chunk is a 60 degree turn), but a stored frame is used only if nothing but rotation/wait was commanded since it was stored, or the place is verified by content (DINOv2-small of the current view against the stored view at the commanded current pose >= 0.75, or against the candidate >= 0.80) |
| Found while building v2: the old rule always takes the 10 nearest poses, however far, so it swaps arbitrary old frames into the window (338 of 611 non-revisit chunks in the full1 stock episodes) | A frame must be within 60 degrees and 2 units of the target pose; otherwise nothing is retrieved and the untouched stock attention runs (pixel-identical to stock) |
| K x 1.28 GB of retrieved K/V held on the GPU (K=10 did not fit an 80 GB card) | Retrieved K/V stay in host RAM and are streamed per attention call; GPU peak 68.9 GB (disk) / 72.4 GB (recompute) against 68.8 GB stock |

## Offline replay (no GPU)

`scripts/memv2_offline.py` on the 91 full1 stock GC and out-of-sight episodes (726 chunks; poses rebuilt from executed
actions, descriptors from the saved videos; 115 chunks revisit a stored view beyond the live window):

| Rule | retrieves at revisit chunks | retrieves at non-revisit chunks |
|---|---:|---:|
| old (10 nearest poses, 24-frame archive) | 115 | 338 |
| v2 verified | 108 (87 rotation-only, 21 content-verified; 7 declined) | 0 |
| v2 content-only | 69 | 129 |

The old archive had already lost the earliest matching frame at 19 of the 115 revisit chunks.

## Step 1: scripted paths (Lambda H100 PCIe, DINO of the final frame against frame 12, one seed)

| Path, scene | stock | old rule | v2 disk | v2 recompute | v2 content-only |
|---|---:|---:|---:|---:|---:|
| turn 204 degrees, hold 6 chunks, turn back; GC007 | 0.352 | 0.780 | 0.816 | 0.765 | 0.585 |
| slow 360 at 30 deg/s; GC007 | 0.568 | 0.486 | 0.630 | 0.615 | 0.528 |
| W x4 then S x4; GC001 | 0.916 | 0.849 | 0.909 | not run | 0.885 |
| hold; GC001 | not back | not back | not back | 0.452 | not back |

Full table with cost: [memory-v2/step1/table.md](memory-v2/step1/table.md); strips in the same folder. 16 of 27 jobs
finished before the GPUs moved to step 2.

- v2 is at or above the old rule on rotation, and at or above stock on the two paths where the old rule hurt.
- Recompute matches disk: the same frames are retrieved on every chunk, final scores within 0.05, and per-chunk DINO
  between the two runs stays above 0.89 on the turn-back chunks. Recompute needs no archive, so it is the primary arm.
- The content-only rule is clearly worse on rotation, so `verified` was carried forward.
- v2 chunks are pixel-identical to stock wherever retrieval does not fire.
- **Hardware caveat.** On the local Thor the stock model itself returns on the GC007 hold path
  (stock 0.798, old rule 0.804, v2 recompute 0.761). "Stock fails the hold" held on Lambda and Modal hardware
  (0.30 to 0.40 on three runs) but is not universal, so the scripted-path gain is one sample per hardware.

Cost on Lambda for a 14-chunk job: recompute 13.5 s in total for 22 recomputed frames, wall 246 s against 176 s stock
(about 6 s per chunk in which retrieval fires, mostly streaming K=10 frames host to GPU); disk 60 to 62 GB written,
8 to 11 s of reads, wall 205 to 311 s.

## Step 2: adaptive revisit episodes (pre-registered)

36 cases (22 GC, 14 out-of-sight) whose full1 budget episode contained a revisit beyond the stock window; v2 recompute
(K=10, native positions) on top of the final1 stock flags, against the final1 stock episode of the same case, seed and
hardware. Gemini (released rubric, `gemini-3.1-pro-preview`), three independent calls per video in both arms.
All 36 completed; retrieval fired in 33.

| Paired v2 minus stock | pairs | mean [95% CI] | W/T/L |
|---|---:|---|---|
| Judge, mean of 3 calls, all | 36 | -0.35 [-0.71, -0.01] | 9/12/15 |
| Judge, held-out cases | 31 | -0.26 [-0.66, +0.10] | 9/11/11 |
| Judge, GC | 22 | -0.45 [-0.95, +0.02] | 8/3/11 |
| Judge, out-of-sight | 14 | -0.20 [-0.67, +0.15] | 1/9/4 |
| Judge, first call only, all | 36 | -0.12 [-0.55, +0.28] | 10/14/12 |
| DINO start frame vs frame at commanded return | 21 | -0.02 [-0.08, +0.06] | 6/6/9 |
| Disk reference arm (exact K/V), judge, 8 held-out GC cases | 8 | +0.32 [-0.17, +0.90] | 4/1/3 |

Files: [per-case.csv](memory-v2-step2/per-case.csv), [report.json](memory-v2-step2/report.json),
[report-first-call.json](memory-v2-step2/report-first-call.json), strips in `memory-v2-step2/strips/`.

The disk reference arm wrote 22 to 62 GB per episode (deleted at episode end), read for 0 to 24 s, and peaked at
69.6 GB of GPU memory; its eight cases are too few to separate from the recompute arm or from stock.

**Reading.** No benefit on real episodes. The judge-independent measure is flat. The judged mean is slightly negative;
the largest losses are episodes where the player ended one chunk earlier in the v2 arm after the first retrieval
(GC036: stock pans back to centre in 8 chunks and scores 5.0 three times; v2 stops after 7 and scores 1.4) or where
the judge disagrees with what is visible (OE069: v2 visibly returns to the opening corridor view, judge 4.25, 1.0, 1.75
against 5.0 three times for stock). Arms differ in player trajectory, so pairs compare episodes, not identical paths.

Why the scripted gain does not transfer:
- Retrieval fires on few chunks (typically 1 to 5 of 6 to 18), usually at or after the return, so the return frame is
  often the same pixels in both arms.
- Most benchmark returns are short, and the start view is already held by the sink frames in both arms.
- Translation returns are not undone by the model; v2 only stops injecting mismatched views.

Examples looked at side by side:
- GC045 (move left 10, right 10; a development case): same nine actions in both arms, first five chunks identical,
  retrieval fired once. Stock's return view has a large traffic light pushed into frame and a changed street sign;
  v2 keeps the Pleasure Pier arch framed as at the start. The cleanest genuine example.
- GC044 (360 in place): v2 ends closer to the opening composition, but the player's last action differs.
- GC022 (Taj Mahal 360): neither arm returns to the opening view; the score gap looks like judge noise.

## Limits

One seed; two scenes on scripted paths; single hardware for step 2; the case rule came from the budget arm's
trajectories on Modal; the disk reference arm ran on 8 cases only and is a sanity check; thresholds were chosen on the
development data listed in the config.
