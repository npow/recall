# Revisit horizon, 24 s away: does an unlimited archive help?

**Status: plan frozen before running (this file and `revisit-horizon-unlimited-jobs.jsonl` are committed before any generation).**

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

## Results

_(filled in after the run)_
