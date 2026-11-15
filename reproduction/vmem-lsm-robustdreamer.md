# VMem, Robust Dreamer, LSM-World: what we can reuse on LingBot-World2

Read 2026-11-14 from the project pages, arXiv HTML and GitHub READMEs. Nothing
here was run; all numbers are the papers' own and unreproduced. Our setting:
LingBot-World2 14B causal-fast (distilled 4-step causal DiT, KV cache, Plucker
camera conditioning, Wan2.2-based, no VACE branch), about 2 hours left, LoRA
allowed, 117 clean Sekai revisit pairs on `worldsync-train-data`, a keyframe-
attention LoRA harness (`worldsync/lingbot_lora.py`, branch
`worktree-agent-aa412da3325e25342`) already training.

## Summary table

| | VMem (ICCV 2025) | Robust Dreamer (arXiv 2605.30855) | LSM-World (arXiv 2606.09828) |
| --- | --- | --- | --- |
| Memory | Surfel-indexed **view** memory. Past views are indexed by the surfels they saw; the target camera renders surfels and retrieves the top-K most-referenced views. | **Latent Gaussian** memory. Generated diffusion latents are stored on Gaussian primitives and recalled by latent-space splatting. | **Latent-attributed 3D points**: one point per latent cell, z-buffer readout of latent tokens plus a visibility mask at the target camera. |
| Geometry needed | Off-the-shelf point maps (CUT3R) on generated views, jointly with retrieved views. Coarse geometry is enough. | CUT3R-initialised ViT-L writer with trained DPT heads for global points and Gaussian attributes (stage 1, about 31 h on 8 H200). | Metric depth (Depth Anything 3), segmentation (SAM3 + Qwen3-VL) to exclude dynamic objects and sky, poses (MapAnything). |
| How it conditions the generator | Retrieved reference views enter as VAE latents plus Plucker embeddings, as extra context frames of an image-set model. | Channel-concat condition `y = concat(X0, X_i, recalled memory latent, control)` into the I2V input. | ControlNet-style side branch mirroring Wan2.2 VACE blocks; segment-aware RoPE marks noisy target, clean preceding, clean reference frames. |
| Base model | SEVA (Stable Virtual Camera). | Wan2.1-I2V-14B-480P. | Wan2.2-TI2V-5B. |
| Adaptation | LoRA rank 256 for an efficient K=4 variant, 600k iterations, 8 A40, lr 3e-6. The K=17 variant uses plain SEVA. | LoRA rank 128, alpha 128, on q,k,v,o,ffn.0,ffn.2; lr 2e-5, 20 epochs; DeepSpeed ZeRO-2; two stages of about 31 h each on 8 H200. | Stage 1: side branch only (lr 1e-5). Stage 2: rank-32 LoRA on q,k,v,o,ffn of self-attention (lr 1e-4). 32 A100, global batch 64. |
| Robustness trick | None. | **Deviation learning**: a Dynamic Deviation Archive stores one-step-approximated rollout deviations by stage and denoising timestamp and injects them into clean training histories (p=0.9). Keeps the first frame clean. | Two-stage training; dynamic-object filter on memory writes. |
| Released | Code (MIT, `runjiali-rl/vmem`), HF model `liguang0115/vmem`, Gradio demo. | **No code link found** in the paper text; a project page exists (not read). | Code and training scripts (MIT repo `microsoft/LatentSpatialMemory`); paper is CC BY-NC-SA. README shows no released checkpoints; you train `vace.safetensors` and `lora.safetensors` yourself. |
| Reported revisit evidence | RealEstate10K long-term (>=200th frame): PSNR 14.09 vs SEVA 13.98, LPIPS 0.452 vs 0.455. Modest. Authors say their cyclic-trajectory protocol is a weak proxy. | ScanNet 300 frames: PSNR 13.43 vs VMem 9.21, SSIM 0.422 vs 0.124. Large gaps; baseline numbers look unusually round, so treat with care. | RealEstate10K closed loop: PSNR_C 20.05 (Spatia 19.38), SSIM_C 0.825 (0.579), LPIPS_C 0.228 (0.213, second-best). |
| Limits stated | Weak evaluation; limited data and compute. | Quality depends on 3D reconstruction; not end-to-end. | Dynamic objects excluded from memory. |

Survey bookkeeping: our earlier survey lists arXiv 2606.09828 as "Mirage:
Latent Spatial Memory". The arXiv page I read is titled "Latent Spatial Memory
for Video World Models" (LSM-World). I annotated that entry rather than
renaming it, and added Robust Dreamer to
`literature-review-2026-11-14.json`.

## Fit to our setting

Common blocker for all three: each needs **real camera geometry** (depth or
point maps) from generated frames, plus a **trained** conditioning path. We
found in `reproduction/realized-motion/` that LingBot realizes rotation at
about 0.9x of commanded but barely realizes forward translation, so a
geometry-based memory has no reliable world-frame to anchor to. LingBot also
has no side branch or channel slot for a rendered memory latent; its I2V
condition is the image latent plus mask, and a distilled 4-step causal model
tolerates little retraining.

### VMem
* **Reusable now (about 60 to 90 min):** the retrieval policy. Our
  `--budget-memory` already retrieves by commanded pose and frustum overlap.
  A surfel-visibility score would help only with translation and occlusion,
  which our current tests do not exercise.
* **Not reusable:** the SEVA conditioning and its LoRA (different generator).
  CUT3R on our frames is an extra model to install and validate; I did not
  check that it runs on aarch64 or Modal images.

### Robust Dreamer
* **Reusable idea, cheap:** deviation learning. Our LoRA harness trains on
  clean ground-truth context. Corrupting the history and memory latents with
  rollout-like deviations (the paper's one-step approximation, or a simpler
  proxy: noise at the sampler's 4 timesteps applied to context latents) can be
  added to `worldsync/lingbot_memory_train.py` context pass in about 30 to 60
  min. It targets the known train/inference mismatch listed in
  `reproduction/train-lora.md` (recomputed vs archived KV).
* **Rewrite:** Latent Gaussian Memory (gsplat, trained DPT heads, 31 h of
  training). No released code found. Not feasible in the time left.
* The base (Wan2.1-I2V-14B-480P) is the closest of the three to LingBot, and
  the LoRA recipe (rank 128, q,k,v,o,ffn) is a useful rank and target
  reference for us.

### LSM-World
* **Reusable idea:** read stored latents back through **latent-space
  reprojection** with a visibility mask. For our **rotation-only** revisits a
  depth-free rotation warp of the stored keyframe latent into the target
  camera is exact up to resolution and borders. That could replace the raw
  keyframe in our existing memory slot. It is an untested hypothesis, not a
  result: the latent grid is coarse (about 30x52) and the model has never seen
  warped latents.
* **Rewrite:** the VACE-style side branch and its two training stages
  (32 A100). The repo has the recipe and scripts for Wan2.2-TI2V-5B, not for
  the A14B MoE we run, and no released weights I could see.

## Ranked recommendation

1. **Add deviation learning to the keyframe-attention LoRA** (Robust Dreamer
   idea). Lowest risk, reuses the harness, directly attacks exposure bias.
   About 30 to 60 min plus one training run.
2. **Test a depth-free rotation warp of the retrieved keyframe latent** as the
   memory-slot content (LSM-World idea), on the existing turn_return and
   turn_hold_return paths with the existing `--budget-memory` plumbing. A
   training-free ablation, about 60 to 90 min. Only valid for rotation
   revisits.
3. **Upgrade retrieval to visibility-aware scoring** (VMem idea) only after
   translation is realized; otherwise it adds little beyond pose-frustum
   overlap.
4. Do not attempt Gaussian memory, a VACE side branch, or CUT3R-in-the-loop in
   the remaining time.

## What I did and did not verify

Verified by reading: the mechanisms, base models, LoRA ranks and targets,
training hardware and durations, and the reported tables above; repo licences
and README commands for VMem and LSM-World (GitHub API and raw README).

Not verified: any reported number (not reproduced); whether Robust Dreamer
has released code (only its paper text was searched; the project page
`hlinchen.github.io/projects/Robust_Dreamer/` was not opened); whether LSM-World
or VMem checkpoints load or run here; whether CUT3R, Depth Anything 3 or
MapAnything run on our Modal image; every "reusable in N minutes" estimate,
which is my judgement and not measured.
