# Memory-keyframe LoRA harness for LingBot-World2 14B causal-fast

Fine-tuning is allowed for this track; this is the training harness, not a result.
Nothing here is a PlayWorld score, and nothing changes stock behaviour by default.

## What it trains

At inference the model attends an 18-latent-frame window: sink (6) + retrieved memory (4)
+ previous chunk (4) + current chunk (4). `worldsync/lingbot_retrieval.py` already builds
that window by swapping the middle four frames for archived KV placed at the virtual slot
`[r-8, r-4)`. The adapter teaches the model to *use* that memory slot when the camera
revisits a view.

`worldsync/lingbot_memory_train.py` reproduces the same window without the stateful KV
cache, so gradients flow:

1. **Context pass (no grad, per layer):** sink, memory and previous chunk are clean (t = 0).
   Each group attends to the earlier groups, at explicit RoPE frame positions
   (sink `0..5`, memory `r-8..r-5`, previous chunk `r-4..r-1`). Produces per-layer K/V.
2. **Target pass (grad, per-block checkpointing):** the revisit chunk `r..r+3` is noised
   with `scheduler.add_noise` at one of the four distilled-sampler timesteps
   (`timesteps_index = [0, 250, 500, 750]`, shift 10) and attends context K/V plus itself.
   Loss is flow matching on that chunk only (`--loss velocity`, MSE to `noise - x0`; or
   `--loss x0`). The sampler is untouched at inference.
3. **LoRA:** rank 32, alpha 32 on q/k/v/o of every block's self- and cross-attention
   (`worldsync/lingbot_lora.py`, no peft). 104.9M trainable fp32 parameters. B is zero-initialised,
   so an untrained adapter is exactly the stock model. Base weights stay bf16 and frozen.

The block math mirrors `CausalWanAttentionBlock.forward` (camera injection, fp32
modulation, cross-attention, FFN). It is checked against the stock KV-cache forward:

* CPU tiny-model test (`tests/test_lingbot_memory_tiny.py`): matches the stock cache path to
  1e-4 in fp32; per-block checkpointing matches no-checkpointing; a LoRA overfit loop reduces loss.
* 14B on GPU: `train_lingbot_memory.py --parity` (result below).

## Data interface

`/data/manifest.jsonl` on the Modal volume `worldsync-train-data`, one JSON object per line:
`{video, start, end, poses, keyframe_time, revisit_time, overlap, [prompt, intrinsics,
wrong_keyframe_start]}`. `video` is an mp4 path under `/data`; `start`/`end` are video-frame
indices; `poses` is an `.npy` of OpenCV camera-to-world 4x4 matrices per video frame; times are
seconds since `start` at 16 FPS, snapped to chunk-aligned latent frames (4 video frames per latent
frame). Requirements checked by `Clip`: the memory chunk lies after the sink and before the
previous chunk, and the revisit chunk is at least latent frame 16 so the virtual memory slot
does not overlap the sink. Frames are resized to 480x832.

Until the real manifest exists, `worldsync/lingbot_memory_data.py` builds *synthetic revisit
clips* from PlayWorld GC images: a pure pan out and back across a mirror-tiled canvas, arranged
so the target chunk shows content that only the memory chunk contains (the sink and previous
chunk do not). They validate the plumbing and measure whether the adapter learns to read memory.
They are not evidence about real revisit quality.

## Running

```bash
MODAL_PROFILE=nissan-pow modal run scripts/modal_lingbot_train.py --steps 40
MODAL_PROFILE=nissan-pow modal run scripts/modal_lingbot_train.py \
    --manifest /data/manifest.jsonl --tasks "" --steps 500
```

Output (`train.json`, `lora.pt`, a 2-chunk `--lora` rollout smoke) lands in the
`worldsync-lingbot-results` volume under `train-lora-<ns>`.

Inference: `worldsync.lingbot_rollout ... --lora PATH`. Because the weights are non-stock it is
refused unless `--controlled-ablation --native-fixed-controls --fixed-actions ...` are all given.
Pair it with `--pose-retrieval` to present memory in the trained window layout.

## Known train/inference gaps

* Context K/V are recomputed in the teacher-forced pass; `--pose-retrieval` archives K/V from the
  original rollout and re-ropes keys at the virtual slot. The adapter's own earlier passes produce
  the archive at inference, so this is close but not identical.
* The sink is one group (stock computes frames 4-5 inside chunk 1).
* Camera translations are normalised over the whole clip; the rollout normalises over the path so far.
* The distilled sampler feeds its own re-noised x0 predictions to later steps; training noises
  ground-truth x0 at each schedule step.
* No real-data results yet; synthetic clips only.

## Results


Synthetic-clip validation run (one Modal H200, 40 steps, train GC001-GC004, held-out GC005; raw
numbers in [lora-validation/synthetic-run.json](lora-validation/synthetic-run.json); adapter at
`worldsync-lingbot-results/train-lora-1794711047856672892/lora.pt`, sha256 `babfd351...f284e0`).
This validates the harness. It is not evidence that revisits improve on real video.

| Check | Result |
| --- | --- |
| 14B teacher-forced forward vs stock KV-cache forward (chunk 1 given clean chunk 0, bf16) | relative mean error 0.67%, max abs 0.37 (bf16 noise; the fp32 tiny-model test matches to 1e-4) |
| Throughput | 6.4 s/step (0.156 steps/s), batch 1 revisit chunk, per-block checkpointing |
| GPU memory | 65.8 GB peak allocated, 72.9 GB reserved of 143 GB; 104.9M trainable fp32 params |
| Wall time | 100 s load, 51 s to encode 5 clips, 769 s total including both evals |
| Save/load | zeroing the adapter restores loss 0.2527; reloading `lora.pt` restores 0.1160 exactly; sha256 matches |
| `--lora` rollout | `lingbot_rollout --lora ... --controlled-ablation` completes 2 chunks, report records rank 32 and the sha |

Flow-matching loss on the revisit chunk, mean over the four schedule steps and two noise draws
(held-out GC005, never trained on):

| Memory in the window | Before | After 40 steps |
| --- | --- | --- |
| correct keyframe | 0.249 | 0.091 |
| wrong keyframe (frames 4-7) | 0.298 | 0.168 |
| no memory slot | 0.292 | 0.167 |

The stock model already prefers the correct keyframe (correct/none = 0.85). After training that
ratio is 0.55 on the held-out clip (0.76 to 0.64 on a trained clip), so reliance on memory
increased. Much of the drop for the wrong/none modes is the adapter learning the synthetic
domain (fast mirror-tiled pans), not memory, which is why the ratio, not the raw drop, is the
signal. Caveats: 40 steps, 4 train clips, one held-out clip, two noise draws, loss on one chunk, no
decoded video, and "wrong" memory overlaps the sink view partly. The training-loss curve in
`train.json` is noisy because each step draws a random schedule timestep; use the eval tables.

## Blockers and notes

* The real manifest (`worldsync-train-data`) did not exist yet; training on it is one command.
* `torch.utils.checkpoint` under CUDA autocast failed on the first Modal run (saved/recomputed tensor
  metadata mismatch on torch 2.6) until the autocast weight cache was disabled
  (`cache_enabled=False`). The CPU and local-GPU tests could not reproduce it.
* H200 queueing, not compute, dominated wall time: the job waited 10-18 minutes for a GPU twice.
* Memory headroom (70 GB) allows larger context (more frames per group) or batch 2.
