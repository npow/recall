# Rotation-only revisit matrix (diagnostic, not a PlayWorld score)

Why: the realized-motion check showed translation barely moves the camera in GC007, so the
`round_trip` path cannot test memory. Rotation is realized at ~0.9 of commanded, so the path is
rotation only, with turn rate 68 deg/s so the realized turn is ~180 degrees.

Paths (fixed actions, native camera controls, seed 42, one Modal H200 each, stock 14B causal-fast):
- `turn_return`: WAIT, RIGHT x3, LEFT x3, WAIT (14 s round trip out of view)
- `turn_hold_return`: WAIT, RIGHT x3, WAIT x6, LEFT x3, WAIT (adds a 6 s absence at the far heading)

Metric: pinned DINOv2-small cosine of the final frame vs frame 12 and frame 0, plus the side-by-side
sheets in `*/comparison.png` (columns: frame 12 | mid, max turn | final). Crude single-seed proxy.

## GC007 (distant facade)

| Method | turn_return f12 / f0 | turn_hold_return f12 / f0 | Peak GB |
| --- | --- | --- | --- |
| stock (18/6) | 0.581 / 0.569 | 0.304 / 0.266 | 80 |
| pose retrieval (existing) | 0.829 / 0.819 | 0.557 / 0.512 | 85 |
| WorldKV-style retention (4 frames, virtual RoPE) | 0.842 / 0.841 | 0.653 / 0.635 | 86 |
| DeCoPrune-style pruning | 0.673 / 0.652 | 0.288 / 0.272 | 85 |
| start-image re-anchor | 0.554 / 0.538 | 0.277 / 0.243 | 85 |
| large cache 46/6 | 0.818 / 0.809 | **0.899 / 0.876** | 115 |

## GC001 (indoor, near-field structure)

| Method | turn_return f12 / f0 | turn_hold_return f12 / f0 |
| --- | --- | --- |
| stock | 0.491 / 0.485 | 0.134 / 0.149 |
| pose retrieval | 0.736 / 0.720 | 0.510 / 0.457 |
| WorldKV-style retention | 0.692 / 0.651 | 0.525 / 0.550 |
| large cache 46/6 | **0.744 / 0.735** | **0.679 / 0.651** |

## Reading

- Visually, stock returns to a different place (GC007: a blue storefront or a cladding close-up;
  GC001: another room). Pose retrieval, WorldKV-style retention and the 46/6 cache bring back the
  original facade or dining room on the short path; on the long absence the 46/6 cache is clearly
  best, WorldKV-style retention is partial, and pose retrieval and DeCoPrune-style pruning fail.
- Re-anchoring and the DeCoPrune-style port did not help here either.
- This reverses the earlier round-trip reading: the "history does not help" result came from a
  path whose translation never realized. With a path that returns, history does help.
- Costs: the 46/6 cache needs 115 GB peak (H200 only, about 1.45x stock time); WorldKV-style
  retention needs a 72 GB CPU archive.
- Limits: one seed per cell, DINO CLS cosine only, two scenes, fixed actions. Differences under
  about 0.05 are not established. No LLM-judged score, no adaptive player, no held-out split.
  Do not read these as PlayWorld scores.
