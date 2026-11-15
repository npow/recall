# DeCoPrune port on the GC007 round trip (diagnostic, not a score)

Port: `worldsync/lingbot_decoprune.py`, spec and deviations: `../decoprune-spec.md`. Path: `round_trip`
(forward 4, turn 180, forward 4, turn 180, hold), 16 chunks, one Modal H200 per run, stock 18/6 window unless
noted. Metric: DINOv2-small CLS cosine, final frame vs frame 12 (the opening facade). **Lower DINO than stock is
worse. None of the final frames is the opening view in any run (see `decoprune-comparison.png`,
`../decoprune-seed43/decoprune-comparison.png`).**

| Run (denoise schedule) | seed 42 | seed 43 |
| --- | --- | --- |
| stock (shift 10, [0,250,500,750]) | 0.608 | 0.571 |
| DeCoPrune, all_bands RoPE (stock schedule) | 0.407 | 0.418 |
| DeCoPrune, RoPE re-index off (stock schedule) | 0.447 | not run |
| stock (CMBench shift 5, [0,179,358,679]) | 0.637 | 0.452 |
| DeCoPrune, all_bands RoPE (CMBench schedule) | 0.600 | 0.515 |

* **Correctness check:** chunks 0-3 are bit-identical to stock in every DeCoPrune run (the first pruning and
  re-indexing happen at chunk 4/5), so history assembly and re-roping match stock where they should.
* **Pruning realised at threshold 0.10:** 60.5% / 59.7% / 61.9% of historical KV (paper: 85.4% on CMBench). Chunks
  1-4, the forward approach to the facade, had every token below threshold (mean MSE ~0.01) and were pruned
  completely; kept tokens came from the turning chunks (kept fractions 0.01-0.33). The method keeps what is
  hard to denoise, which here is not the content a revisit needs.
* **Stock schedule:** DeCoPrune is worse on both seeds (-0.20, -0.15 DINO). RoPE re-indexing off is less bad than on.
* **CMBench schedule:** direction flips with the seed (-0.04 on 42, +0.06 on 43) and stock itself moves 0.64 -> 0.45
  between seeds. Two seeds cannot separate method from noise.
* **Cost:** peak GPU 99-103 GB vs 80 GB stock; wall time about the same (rebuilt attention each call).
* **Not shown:** any benefit of DeCoPrune on this task. Not tested: other thresholds, step index, per-head variant,
  observed-prefix scoring, CMBench itself.
