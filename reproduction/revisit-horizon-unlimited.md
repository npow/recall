# Revisit horizon, 24 s away: does an unlimited archive help?

**Status: partial pilot done (8 of 20 scenes). Plan and job list were committed before any generation (882a3dc).**

Question. The bounded memory behind the revisit-horizon result keeps at most 24 latent frames (about 6 s of video, evicting the
most pose-redundant frame when full). Its gain shrank to +0.34 [+0.05, +0.64] at 24 s away. Does removing the cap improve on that?

## Design

- **Scenes, path, seed, judge:** identical to [revisit-horizon.md](revisit-horizon.md): the same 20 held-out GC scenes, the
  scripted 24 s look-away (hold 1 s, turn right about 180 degrees, wait 24 s, turn back, hold), seed 42, fixed actions, and the
  released PlayWorld verifier with 2 Gemini calls per video (`gemini-3.1-pro-preview` on Vertex).
- **New arm `memory_unlimited`:** the 24 s memory job of each scene with one change, `--budget-archive 128`
  (24 s away is 32 chunks = 128 latent frames, so nothing is ever evicted). Exact storage, nearest commanded-pose retrieval,
  `--budget-retrieved 10`, native RoPE: everything else identical to the 24-frame arm.
- **Reused arms:** stock and 24-frame memory results at 24 s from the main run (not rerun).
- **Hardware caveat:** the reused arms ran on Lambda H100 PCIe (and 16 pairs on the 4-GPU SXM5 box); the new arm runs on whatever
  was available at the time (see Results). Outputs are not bit-identical across GPU models, so a small hardware effect is possible.
- **Analysis:** paired by scene, bootstrap 95% CI over scenes (same code as the main analysis). Compared against the 24-frame arm
  (primary) and against stock (secondary). A null result will be reported as a null.
- **Scope:** 24 s only. If the run does not finish all 20 scenes, the result is reported for the scenes that did, and says so.

## Results (partial pilot: 8 of 20 scenes)

**The unlimited archive scored higher than the 24-frame archive on the 8 scenes that finished, but this is a small partial run
and the intervals are wide.** It is not a replacement for the 20-scene result.

| 24 s away, 8 scenes | mean score (1-5) | paired difference vs unlimited [95% CI over scenes] | W / T / L |
|---|---:|---|---|
| stock | 1.41 | unlimited minus stock: +1.28 [+0.70, +1.84] | 7 / 1 / 0 |
| 24-frame memory | 1.64 | unlimited minus 24-frame: **+1.05 [+0.51, +1.50]** | 6 / 0 / 2 |
| **unlimited archive** | **2.69** | | |

Per scene (mean of 2 judge calls; unlimited / 24-frame / stock): GC042 1.2 / 1.4 / 1.0, GC014 3.8 / 2.6 / 1.1, GC032 3.0 / 1.5 / 1.3,
GC019 1.1 / 1.2 / 1.1, GC011 2.9 / 1.7 / 1.7, GC034 4.0 / 2.4 / 2.1, GC002 3.0 / 1.3 / 1.5, GC035 2.5 / 1.0 / 1.5.
DINO final-frame similarity to the start view: +0.02 [-0.08, +0.11] vs 24-frame (no difference), as in the main benchmark.

Which scenes. The job list order is the frozen order. These are the first 7 scenes of that list (GC042, GC014, GC032, GC019, GC011, GC034,
GC002) plus the 9th (GC035); the 8th (GC044) had not finished when the run was stopped, so the set is not strictly "the first 8".
No scene was chosen or dropped by its score. On the first 7 only: +0.99 [+0.43, +1.50] vs 24-frame (5 / 0 / 2), +1.31 [+0.67, +1.97] vs stock.

## Caveats

- **Partial.** 8 of 20 scenes, one seed. The 24-frame memory mean on these 8 scenes is 1.64 against 1.79 on all 20, so these scenes
  were slightly harder for it; the unlimited arm's mean of 2.69 is not an estimate for all 20.
- **Judge noise is large.** Two calls on the same video differ a lot (GC032: 5.0 and 1.0; GC034: 3.0 and 5.0). Intervals are over scenes,
  with 2 calls per video.
- **Hardware.** The new arm ran on a single H100 SXM5 and a single H100 PCIe (Lambda). The stock and 24-frame arms ran on H100 PCIe and
  on a 4-GPU SXM5 box. Outputs are not bit-identical across GPU models, so a small hardware effect cannot be excluded.
- **Not reproduced elsewhere.** No confirmation run, and 12 scenes untested.
- **Why only 8.** The first box (4x H100 SXM5) was unusable (NVSwitch fabric manager failure, CUDA error 802) and was terminated; only two
  single-GPU boxes could be obtained afterwards.

## Reproduce

Jobs: `revisit-horizon-unlimited-jobs.jsonl` (frozen in 882a3dc before any generation, `--budget-archive 128`, everything else as the 24-frame arm).
Scoring and analysis: `python scripts/revisit_horizon_unlimited.py score` then `analyze`; per-scene numbers in `revisit-horizon-unlimited/analysis.json`.
