# Bounded pose-indexed KV memory (diagnostic, not a PlayWorld score)

`worldsync/lingbot_budgetmem.py`, flag `--budget-memory` (`--budget-retrieved K`, `--budget-archive N`,
`--budget-rope virtual|native`). Fixes the three audited problems in `lingbot_kvretention.py`:
selection uses commanded pose (position + heading) against the chunk's target pose instead of a
content-blind query from the noisiest denoising step; the gate derives from the real window; the archive
is bounded (N latent frames, pose-redundancy eviction, canonical bf16 K/V on CPU, ~1.3 GB per frame).
Attended window = sink 6 + K retrieved + tail 8.

GC007, rotation-only paths, 68 deg/s turns, DINOv2-small cosine of the final frame vs frame 12 / frame 0.
Stock and 46/6 seed 42 come from `reproduction/revisit/`; seed 43 baselines were rerun here. Budget runs are seed 42 only.

| turn_return | f12 | f0 | peak GB | s |
| --- | --- | --- | --- | --- |
| stock s42 / s43 | 0.581 / 0.700 | 0.569 / 0.686 | 80 | 147 / 139 |
| 46/6 s42 / s43 | 0.818 / 0.818 | 0.809 / 0.805 | 115 | 147 / 152 |
| budget K6 virtual | 0.840 | 0.827 | 88 | 175 |
| budget K10 virtual | 0.818 | 0.807 | 93 | 177 |
| budget K10 native | 0.773 | 0.772 | 93 | 165 |
| budget K14 virtual | 0.829 | 0.817 | 97 | 176 |

| turn_hold_return | f12 | f0 | peak GB | s |
| --- | --- | --- | --- | --- |
| stock s42 / s43 | 0.304 / 0.404 | 0.266 / 0.403 | 80 | 181 / 178 |
| 46/6 s42 / s43 | 0.899 / 0.878 | 0.876 / 0.878 | 115 | 212 / 194 |
| budget K6 virtual | 0.558 | 0.509 | 88 | 245 |
| budget K10 virtual | 0.713 | 0.706 | 93 | 248 |
| budget K10 native | 0.768 | 0.751 | 93 | 262 |
| budget K14 virtual | 0.629 | 0.625 | 98 | 239 |

Images: `turn_return/comparison.png`, `turn_hold_return/comparison.png` (frame 12, max-turn frame, final).
Visually, on the long-hold path K10 native, K10 virtual and K14 virtual all return to the Primemall facade
(stock returns to an interior; K6 returns to a close-up of the facade sign); 46/6 returns most faithfully.

Caveats: one scene (GC007), one seed per budget cell, DINO CLS cosine is a crude proxy (K14 virtual
scores 0.63 yet visibly returns to the facade), the stock/46/6 baselines move 0.1 between seeds 42 and 43
on the same path, so gaps under ~0.1 are not established. GC001 was not run. Peak GPU is 88-98 GB, so the
90 GB target is met only by K6 (which fails the long hold); CPU archive stays below ~31 GB (24 frames).
Pose retrieval uses commanded poses, which are valid only on rotation paths where the model realizes
motion; translation is barely realized (see `reproduction/realized-motion/`).
