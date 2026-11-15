# DeCoPrune-style token pruning on the GC007 round trip (diagnostic)

GC007, seed 42, fixed `round_trip` path (16 chunks), one Modal H200 each. DINOv2-small
cosine between the reference frame (12) and the final frame; stock = 0.612.

| Run | RoPE | Schedule | DINO | Retention | Peak GB |
| --- | --- | --- | --- | --- | --- |
| stock | native | stock (shift 10) | 0.612 | n/a | 80 |
| DeCoPrune-style | compact | stock | 0.589 | 12.0% | 87 |
| DeCoPrune-style | native | stock | 0.513 | 11.8% | 87 |
| DeCoPrune-style | compact | DeCoPrune (idx 0,179,358,679, shift 5) | 0.543 | 6.4% | 84 |

No variant restores the start view (`comparison.png`); all end in the plaza.
The dpsched row has its own frame-12 reference (different schedule); vs the stock
reference it is 0.541. Single seed, single path, one crude metric: differences of
~0.05 are not established. Scoring/probe/threshold follow the released CMBench
`decoprune` config (probe step 2, threshold 0.10, MSE per 2x2 patch); the cache
mechanics, RoPE re-index ("all_bands" interpreted as compact temporal positions) and
the lack of an observed-prefix scoring pass are documented deviations in
`worldsync/lingbot_decoprune.py`.
