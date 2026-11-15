# Full-history cache pilot

GC007, seed42, fixed native turn-and-return path, 7.8125 simulated seconds. Increase cache capacity from18 to34 latent frames (sink6 unchanged); this retains the entire32-frame history. No training, image editing, or controller changes.

The native camera input audit passes for both. The first4 chunks have identical raw pixel hashes; divergence starts when stock evicts history. Endpoint DINOv2-small cosine rises from0.5788 to0.8481. PSNR falls from9.263 to9.117dB: exact pixel reconstruction is not improved. Blinded full-video Claude Opus5.5 comparison prefers candidate A with high confidence: it recovers the original roof, facade and sign through continuous reverse motion, while stock invents a different roofline/sign. Both still hallucinate intermediate scenery, and candidate has framing drift and an added foreground umbrella. This is diagnostic evidence, not a PlayWorld score.

Peak allocated GPU memory rises from80.20 to99.92GB (+24.6%). Seven warmed generation intervals take36.89s stock and40.70s candidate (+10.3%); total pipeline times include loading and are not inference latency. Timing uses one run per mode and has no uncertainty estimate.

Videos, contact sheets, complete run manifests, and blinded comparison are committed here. Settings are frozen for the predeclared additional seed/scene pairs in ../context-validation-plan.json. No general improvement or SOTA claim until validation.
