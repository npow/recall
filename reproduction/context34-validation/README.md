# Frozen context validation

Increasing the native cache from 18 to 34 latent frames improves returning landmark identity on two scenes, including an image-held-out scene. It fails to fix the windmill. Settings were frozen before these three validation pairs were generated; the initial pilot is shown separately below.

| Scene | Seed | Stock DINO | Cache34 DINO | Return recovered, stock → candidate |
|---|---:|---:|---:|---|
| Mall GC007, pilot | 42 | 0.579 | 0.848 | no → yes |
| Mall GC007, new seed | 43 | 0.680 | 0.814 | no → yes |
| Windmill GC004, development | 42 | 0.727 | 0.678 | no → no |
| Museum GC002, held out | 42 | 0.665 | 0.861 | no → yes |

DINOv2-small normalized CLS cosine compares frame12 to the final frame. All four pairs complete the same 7.8125-second native camera path: hold, right turn, left turn, hold. The native camera tensor audit passes, and raw pixels match for the first four chunks until cache eviction differs. Weights, source, sampler, seed, resolution and actions are matched within each pair. Full-video blind comparisons assign high confidence to the mall and museum improvements; the windmill preference is weak and does not count as a recovered return. Anonymized label order is fixed by a hash of scene/seed for the validation cases.

This is evidence that discarded view history causes some return failures. It does not establish exact 180-degree realized rotation, exact pose recovery, a minute-long persistent world, or an adaptive PlayWorld score. All short paths fit in the larger cache, so a separately declared delayed-return stress test exceeds both cache windows. The context knob remains opt-in and adaptive tuning remains gated.

Failures remain: foreground objects can appear, framing drifts, geometry morphs, and the windmill overshoots into an invented mill. Pixel PSNR gets worse on both mall seeds and the windmill, despite improved mall identity. The museum improves both PSNR and DINO. The prior retrieval and sink-key rephasing attempts are rejected and preserved elsewhere.

Peak GPU allocation is about 80.2 GB stock versus 99.9 GB candidate. Timing reports exclude loading by measuring intervals after chunk0; these single-run timings are exploratory, with no uncertainty estimates. See each comparison.json for full timing, metrics, judge evidence and label assignment; each directory includes videos, endpoint PNGs, contact sheets and run manifests. The held-out museum case has now been consumed for evaluation and must not become a tuning case.

Reproduce generation with `modal run --profile nissan-pow scripts/modal_lingbot_roundtrip.py --validation`. Evaluate all declared pairs with `.venv-lingbot/bin/python scripts/evaluate_context_validation.py --judge`. The original plan is ../context-validation-plan.json. The full benchmark's published scores use a different evaluator and hosted model; these results are not comparable to Genie 3's leaderboard score.
