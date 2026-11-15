# Full-history cache pilot

GC007, seed 42, fixed native turn-and-return path, 7.8125 simulated seconds. Increase cache capacity from 18 to 34 latent frames (sink 6 unchanged); this retains the entire 32-frame history. No training, image editing, or controller changes.

The native camera input audit passes for both. The first 4 chunks have identical raw pixel hashes; divergence starts when stock evicts history. Endpoint DINOv2-small cosine rises from 0.5788 to 0.8481. PSNR falls from 9.263 to 9.117 dB: exact pixel reconstruction is not improved. Blinded full-video Claude Opus 5.5 comparison prefers candidate A with high confidence: it recovers the original roof, facade and sign through continuous reverse motion, while stock invents a different roofline/sign. Both still hallucinate intermediate scenery, and candidate has framing drift and an added foreground umbrella. This is diagnostic evidence, not a PlayWorld score.

Peak allocated GPU memory rises from 80.20 to 99.92 GB (+24.6%). Seven warmed generation intervals take 37.58 s stock and 40.70 s candidate (+8.3%); total pipeline times include loading and are not inference latency. Timing uses one run per mode and has no uncertainty estimate.

Videos, contact sheets, complete run manifests, and blinded comparison are committed here. Settings are frozen for the predeclared additional seed/scene pairs in ../context-validation-plan.json. No general improvement or SOTA claim until validation.
