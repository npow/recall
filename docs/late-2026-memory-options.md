# Recent memory options: release check on November 15, 2026

This is an update to the experiment choices, not an exhaustive literature review or a score reproduction. Paper results use different tasks, controls and judges; none of the numbers below establishes a 2.12 PlayWorld overall.

| Approach | Mechanism | Practical status for this work |
|---|---|---|
| Commit Forcing, Oct 31 | Restrict clean cache-writing attention to the preceding block plus current self-tokens; retain full history for denoising | Isolated LingBot transfer implemented with native CPU tests; GPU comparison being prepared. No training needed. |
| History Forcing, Nov 2 | Train on generated histories using the difference between teacher predictions with and without informative history | Public code/checkpoints. Its world-model release uses DreamX5B, so it needs a separate baseline and control integration. |
| World-T3, Oct 12 | Pose-keyed fast-weight memory updated by recursive least squares | Project page demonstrates the method but does not link a code or weight release in this check. Not a verified runnable replacement here. |
| NanoKV, Nov4 | Released fitted LingBot-World2 KV codec with frozen generator | Closest direct compression test after native58 recovered initial landmarks. Artifact verified; identity-parity and longer logical-cache quality still pending. [Source](https://github.com/jiacheng-lu03/NanoKV). |
| L3G-World, Oct 29 | Persistent3D Gaussians carrying VAE features, rendered directly into diffusion conditioning | Official repository still lists inference/checkpoints as an unchecked release plan. Rebuilding its learned components would exceed this track's constraints and deadline. |

## What can transfer immediately

[Commit Forcing](https://arxiv.org/html/2611.00937v1) separates denoising context from the context used to write persistent KV. It retains current-block self-tokens during writing; the first block is self-only. For LingBot's four-latent blocks, the proposed clean read contains up to eight frames. The paper evaluates other generators, so our transfer still needs a matched GPU test and motion/revisit inspection. It is an alternative ablation, separate from retrieval and the learned reader.

## What would require a new training/evaluation setup

[History Forcing](https://arxiv.org/html/2611.02782v1) uses on-policy student histories and fixed-camera teacher evolution for out-of-sight supervision. This is relevant to the gap between our teacher-forced denoising loss and generated-world behavior. Its world-model experiment trains on 200 Pexels images with four camera trajectories each, then evaluates 43 PlayWorld out-of-sight cases with preset trajectories. Crucially, its 2.49 gated mean uses **Doubao Seed2.0 Lite**, rather than our released Gemini setup, and is a single dimension rather than a four-dimension overall. We cannot call that a Genie 3 win. [Released code and checkpoint instructions](https://github.com/ziplab/History-Forcing).

## Why the attractive alternatives are not active implementations

The [World-T3 project](https://adreamwu.github.io/World-T3/) describes pose-conditioned test-time memory. Absence of linked release artifacts on that page is narrower than proving that no release exists anywhere; we have not verified a runnable checkpoint.

The [L3G-World repository](https://github.com/cvlab-kaist/L3G-World) currently contains the release plan, assets and notices. Its learned Gaussian updater/generator are not drop-in renderers for frozen LingBot. Training those replacements ourselves would be a new project, rather than a bounded inference or adapter experiment for this submission.

## Integration checks after source inspection

History Forcing code revision `9d69c1e5da2beb219e635f4400b8cb494759ba06` contains an additional old LingBot backend, but its published world-model recipe/checkpoint targets DreamX5B. The LingBot path uses three-frame blocks and different temporal/first-latent handling from v2; its inference helper also references undefined `tpf`. It needs explicit scheduler, cache, conditioning and zero-adapter parity work before training. The positive native58 control makes a verified compression experiment a more direct next step. [NanoKV integration evidence](../reproduction/nanokv/README.md).
