> **Integration note (main).** Only the analysis and manifest scripts, the GPU probe and this document are on `main`.
> The episode runner `scripts/modal_lingbot_powered.py` and the rollout flags it passes (`--end-guard`, `--player-retries`,
> `--experimental-adaptive-ablation`, adaptive `--budget-memory`) are **not on main**: they conflict with main's own player v6
> and hillclimb unlock, so the runner would fail on main as it stands. They live on branch
> `worktree-agent-a5f22f4098cb528aa` (commits 2b2d49e, 645b271). Reconcile them with main's hillclimb mechanism before running.

# Powered paired judged comparison: stock vs bounded pose memory vs 46/6 cache

**Status: BLOCKED before any generation. No scores exist.**

The `krasnaya-zvesda` and `seraphina-ink-88` Modal workspaces refuse every GPU type:
`InvalidError: Please add a payment method to use <H200|H100|A100-80GB|B200> GPU functions.`
(probed with `scripts/modal_gpu_probe.py`; the volume seeding in `krasnaya-zvesda` is CPU-only and worked).
Only `nissan-pow` can launch GPUs, and this task was told not to use it.

## What is ready (committed, tests pass: 15 in plan_guard/tuning/budgetmem)

* Merged branches: cohort runner + `--end-guard` + adaptive ablation (a09db12), bounded pose memory
  (`worldsync/lingbot_budgetmem.py`, wired into the rollout, a4e995c, re-integrated by hand because the
  branch's `lingbot_rollout.py` conflicted in 5 places), Gemini Vertex judge (a0a0580).
* `--budget-memory` now also works in adaptive episodes when combined with `--experimental-adaptive-ablation`
  (the rollout already calls `memory.begin(index, camera.poses)` after each player decision, so the pose list
  is available online). Stock knobs are required. It releases temporary GPU reads before VAE decode.
* `--player-retries N` (default 0): resample the player on `PlayerResponseError`; recorded in `run.json`.
  Deviation from the reconstructed protocol, applied to every arm equally.
* `scripts/modal_lingbot_powered.py`: 14 cases x seeds x arms {stock, budget, cache46}; every arm gets
  `--end-guard --player-retries 2`; the judge sees a neutral world-model name (no arm label); Opus scores in Modal.
* `scripts/build_powered_manifest.py` + `worldsync.gemini_vertex_judge` (local, gcloud login) for the Gemini score;
  `scripts/analyze_powered.py` for per-arm means, paired deltas with bootstrap CIs over (case, seed), GC/OE splits,
  Gemini category means and generation-failure counts.

## To run once a GPU-capable workspace exists

```
MODAL_IMAGE_BUILDER_VERSION=2025.06 MODAL_PROFILE=<gpu workspace> modal run scripts/modal_lingbot_powered.py --seed 42   # then 43, 44
# download rollout.mp4, run.json, judge.json for each powered-* episode, then:
python scripts/build_powered_manifest.py <episodes dir> <out>
python -m worldsync.gemini_vertex_judge --manifest <out>/manifest.jsonl --output-dir <out>/episodes
python scripts/analyze_powered.py --episodes <out>/episodes --meta <out>/meta.json --generated outputs/lingbot-reproduction/powered-seed*/generated.json --output <out>
```

The new workspaces need the weights volume seeded first (`scripts/modal_seed_weights.py`; `krasnaya-zvesda` is done).
Budget: 14 cases x 3 seeds x 3 arms = 126 episodes, roughly 8 H200-minutes each, about 17 GPU-hours.

## Caveats for whoever runs it

* The budget arm retrieves by commanded pose, so it only matters when the episode revisits a pose; the earlier
  guarded cohort found most reference plans short. The 14 cases were picked by plan duration, not outcome.
* The Gemini judge agrees only weakly with Opus (Spearman about 0.32 in the earlier rescoring); report Gemini as primary
  and Opus as secondary, never mix.
* Nothing here is comparable with the paper's 1.82 / 2.12 beyond using the released rubric on a different
  model checkpoint, player and video length.
