# Memory LoRA on real revisit data: first result (negative on quality)

Diagnostic on fixed native-camera paths, one Modal H200 per run, DINOv2-small cosine of the final frame
against frame 12 (the start view). Not a PlayWorld score and not judged. Table: `table.md`, `table.json`;
sheets: `<scene>-<path>-seed<N>/comparison.png`; degradation: `degradation.json`.

## What was trained

* Harness: `worldsync/lingbot_lora.py`, `lingbot_memory_train.py` (rank 32 on self/cross-attention q/k/v/o,
  flow-matching loss at the four distilled-sampler timesteps, memory keyframe at the virtual slot).
* Data: 48 Sekai revisit pairs (`content_ok`, chunk-aligned indices valid, 10 held-out pairs from different
  source videos). 8 more pairs were dropped by index constraints (keyframe before latent frame 8).
* Run: 200 steps, lr 1e-4, batch 1, 6.3 s/step, 84.7 GB peak allocated, 3139 s wall incl. 668 s encoding.
  Adapter `worldsync-lingbot-results/train-lora-1794712363707558680/lora.pt` (sha256 `ee5113a4...c1b345f`).

| Held-out flow-matching loss (10 pairs, 11 clips counted) | Before | After 200 steps |
| --- | --- | --- |
| correct keyframe in memory slot | 0.0958 | 0.0540 |
| no memory slot | 0.0896 | 0.0565 |

The stock model did not prefer the right keyframe on real video (correct/none 1.07); after training the ratio is
0.96. Most of the drop is domain adaptation; reliance on memory barely moved.

## Revisit results (DINO end-vs-frame12, seeds 42 and 43; all arms at the stock 18/6 window except ctx46)

| scene/path | stock | retrieval | **retrieval + LoRA** | 46/6 cache |
| --- | --- | --- | --- | --- |
| GC007 turn_return | 0.641 | 0.788 | 0.801 | 0.818 |
| GC007 turn_hold_return | 0.352 | 0.590 | 0.630 | 0.890 |
| GC001 turn_return | 0.477 | 0.672 | **0.460** | 0.637 |
| GC001 turn_hold_return | 0.264 | 0.515 | 0.596 | 0.663 |

* Per-seed values and memory/time are in `table.md`. Seed noise is 0.05-0.25 for stock on these paths.
* The adapter adds +0.01 to +0.08 over plain pose retrieval on three cells (inside seed noise) and loses 0.21 on
  GC001 turn_return (0.561 and 0.360 against 0.745 and 0.598). Bounded retrieval alone already captures most of the
  gain over stock, but the 46/6 cache still wins on the long absence (GC007 hold 0.890 vs 0.630).
* Peak memory with retrieval plus LoRA is 86 GB, 1.0x stock time, versus 115 GB and 1.45x for 46/6.

## It degrades ordinary generation

On `turn_away` (turn 180 degrees and never return; no memory is used):

| scene | arm | DINO vs stock (mean over 8 frames) | mean abs pixel diff (0-255) |
| --- | --- | --- | --- |
| GC007 | retrieval | 0.999 | 1.1 |
| GC007 | retrieval + LoRA | 0.606 | 38.1 |
| GC001 | retrieval | 0.998 | 1.2 |
| GC001 | retrieval + LoRA | 0.507 | 42.2 |

`GC007-turn_away-seed42/comparison.png` shows the adapter's turned view is smeared and washed out compared with
stock's sharp storefront, so the adapter changed generation well beyond the memory slot. The higher end-vs-f12
number on that scene (0.515 vs 0.424) is not a quality gain.

## Conclusion and caveats

* This first adapter does not beat plain bounded pose retrieval and damages normal generation; it does not
  approach the 46/6 cache on long absences. Do not use it.
* Likely causes (untested): 200 steps at lr 1e-4 on 48 clips with a distilled few-step model and a global rank-32
  adapter on cross-attention too; train/inference gaps listed in `reproduction/train-lora.md`; mostly 8-12 s gaps and
  no captions in the data.
* Single adapter, two seeds, two scenes, fixed paths, DINO only, no judged score. A fair retry would train on
  self-attention only with lower lr or a smaller rank, add an explicit no-memory regularizer (match stock outputs on
  no-revisit clips), and mine more and longer revisits.
