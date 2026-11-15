# Integration of parallel fork experiments (2026-11-14)

This records what the parallel experiment forks found, where each result and piece of code now lives on `main`,
and what stayed on a branch. Everything here is a **local-judge or fixed-path diagnostic**. None of it is a
reproduction of the published PlayWorld numbers (LingBot-World2 1.82, Genie 3 2.12).

## Headline, with the caveat that matters

* **Fixed revisit paths (DINO similarity of the final frame vs the start view): memory helps a lot on long absences.**
  GC007 `turn_hold_return`, mean of seeds 42/43: stock 0.35, bounded pose retrieval 0.59, 46/6 cache 0.89.
  GC001 hold: stock 0.26, retrieval 0.52, 46/6 0.66. Best bounded variant (budget memory K10 native) reached 0.77 on
  GC007 hold at about 93 GB, against 115 GB for 46/6.
* **Judged adaptive episodes: no effect shown.** 46/6 vs stock, 15 development cases, seeds 42 and 43, `--end-guard`, local Opus
  judge: mean 2.19 vs 2.23, paired delta +0.09 (9 wins, 10 ties, 5 losses), well inside seed noise (identical settings
  swing up to about 2 points). Most reference plans are short (2 to 4 s of absence), which fits inside the stock window,
  so the benchmark rarely forces the regime where memory matters.
* **Do not read the DINO gains as a PlayWorld score gain.** DINO is a crude proxy, the paths are scripted, scenes are two,
  and stock itself moves about 0.1 between seeds.

## Where each finding lives

| Finding | Evidence on main | Code on main | Branch / commit |
| --- | --- | --- | --- |
| Larger cache alone fixes revisits on valid paths (not on the old round trip, whose translation never realised) | `reproduction/cache-diagnosis/`, `reproduction/revisit/` | tuning flags already on main | `a59918461` 08ac45f, `ac2ad4cab` e245579 |
| Start-view re-anchoring: keeps sign/identity, not the framing; no pixel gain | `reproduction/revisit/` (reanchor rows) | `worldsync/lingbot_reanchor.py`, `--reanchor` | `afb160287` 63f4d6b |
| WorldKV-style KV retention: partial gains, three audited defects | `reproduction/kv-retention/` | `worldsync/lingbot_kvretention.py`, `--kv-retention` | `ab14e26c0` cb60021 |
| DeCoPrune (source-faithful port): worse than stock on this path; prunes the forward-approach chunks | `reproduction/decoprune-spec.md`, `reproduction/decoprune/`, `decoprune-seed43/` | `worldsync/lingbot_decoprune.py`, `--decoprune`, `--denoise-schedule` | `aeede8232` c0359c7 |
| DeCoPrune-style first port (token-level, compact RoPE positions, built on the KV-retention plumbing; no CMBench cache) | `reproduction/decoprune-a/` (results only) | not merged (superseded by the source-faithful port) | `a51b63e0a` 85a5f0f |
| Bounded pose-indexed memory (sink + K retrieved + tail, evicting archive) approaches 46/6 at lower memory | `reproduction/budgetmem/` | `worldsync/lingbot_budgetmem.py`, `--budget-memory` | `a4e995c11` 56cb122 |
| Rotation realised at about 0.9x commanded; forward translation depends on scene distance | `reproduction/realized-motion/`, `reproduction/prompt-motion/` | `scripts/realized_motion.py` and companions | `a1110f346` 2bb0503, `ab2822042` c552398 |
| Text prompts and translation gain do not raise forward motion | `reproduction/prompt-motion/` | `worldsync/lingbot_prompt_motion.py` (module and tests only) | `ab2822042` c552398 |
| Judged scores are driven by trajectory realisation and short plans, not forgetting | `reproduction/score-drivers.md`, `reproduction/cohort/` | `worldsync/plan_guard.py` (module only) | `a09db1298` c2b1f7a, `a5112f101` 09152eb |
| Measured-motion feedback to the player: +0.37 over 9 pairs (not established); turn compensation hurt; more invalid-action failures | `reproduction/cohort/README-feedback.md` | `worldsync/motion_feedback.py` (module and tests only) | `a4a049746` bbbf393 |
| Memory LoRA (rank 32, 48 Sekai pairs): no gain over pose retrieval, degrades ordinary generation (negative result) | `reproduction/lora/`, `reproduction/lora-validation/`, `reproduction/train-lora.md` | `worldsync/lingbot_lora.py`, `lingbot_memory_train.py`, `lingbot_memory_data.py`, train scripts | `aa412da33` 0a954a5, `a022e5e04` 2bc10e6 |
| Sekai revisit-pair data pipeline (117 clean pairs) | `reproduction/train-data.md` | `worldsync/revisit_pairs.py`, `scripts/modal_build_revisit_data.py` | `afc3de966` 24c2400 |
| Persistent Modal runner: 3.6x faster per job, bit-identical to one-shot on the same GPU model | `reproduction/modal-runner.md`, `reproduction/runner-validation/` | `scripts/modal_lingbot_runner.py`, `lingbot_runner_client.py`, `worldsync/lingbot_runtime.py`, `lingbot_runner_core.py`; rollout `main(argv, runtime)` and `--lora` (merged later, see below) | `abf23adbd` 020259a |
| Gemini judge via Vertex AI (gcloud login) | `worldsync/gemini_vertex_judge.py` | yes, no credentials stored | `a0a0580ab` f52cc91 |

## What is wired and what is not

Wired into `worldsync/lingbot_rollout.py` as separate, default-off, mutually exclusive diagnostic flags (native fixed paths
only): `--kv-retention`, `--decoprune` (+ `--denoise-schedule cmbench`), `--reanchor`, `--budget-memory`. Stock defaults,
gating and the stock sampler arguments are unchanged. `PoseKVMemory` gained an `archive_range` attribute (default
`(6, 22)`, so behaviour is unchanged) which re-anchoring needs.

**Tested here:** unit tests (119 pass, 4 skip for the ABot setup), the CLI `--dry-run` for every new flag, and the mutual-exclusion
error. **Not tested here:** a GPU run of the new wiring on main's newer `PoseKVMemory`. The fixed-path numbers above were measured
on the older base the forks started from. The subclasses reuse the same bank and attention hooks, but treat them as
unverified on main until a short GPU smoke run reproduces a branch result.

Left on branches (the forks' code conflicts with main's newer player and runner, or needs more than a mechanical merge):

* `--end-guard`, measured-motion feedback and `--compensate-turn-rate` wiring into the player and rollout. Main's Player v6
  already has an END challenge and a hillclimb subset; reconcile before porting. The player-prompt test for the feedback hint
  was left out for the same reason.
* `--motion-prompt` and `--translation-gain` flags in the rollout (`--lora` is now wired, see the runner section), and the LoRA evaluation scripts that need the first two.
* `scripts/launch_revisit_matrix.sh` and the older `modal_lingbot_roundtrip.py` option set the revisit matrix used.
* Two forks were still running when this was written and are not integrated: the powered judged comparison on a second Modal
  workspace and the near-field motion retest on a third.

## Audit notes that must travel with the code

* **KV retention** (`worldsync/lingbot_kvretention.py`): an independent audit confirmed three real defects. Selection is made once per
  layer on the first, highest-noise attention call, so the query is content-blind in early layers; the gate hard-codes the
  18/6 window, so other windows silently run stock attention; and the archive is unbounded full-precision per-layer KV on CPU,
  about 1.3 GB per latent frame (roughly 200 GB over 40 chunks). The bounded `--budget-memory` method fixes these.
* **DeCoPrune port**: deviates from the released runtime (no observed-prefix probe; real RoPE kept for q0 <= 17;
  attention rebuilt per layer from our own archive; threshold not re-tuned). Two seeds only. Seed noise on stock under the
  alternative schedule moved 0.64 to 0.45.
* **Memory LoRA**: trained on 48 pairs for 200 steps; held-out loss fell about the same with and without the memory slot, so it
  mostly adapted to the domain. It drifts ordinary generation (DINO vs stock 0.51 to 0.61 on a no-revisit path).

## Caveats

* Scores from our local Opus judge are not comparable with the paper's. A Gemini rescoring was in progress and shows only weak
  per-episode agreement with Opus (Spearman about 0.32 over 44 episodes at the time of writing).
* One or two seeds per cell, two scenes for fixed-path tests, DINO final-frame similarity as the metric.
* The camera does not return to the same place on the original round-trip path (translation is scene-dependent), so memory
  results use rotation-only revisit paths.
* No secrets, tokens, weights, videos or per-frame images are committed; per-frame PNGs and results over a few MB were left out.

## Corrections and warnings added after the near-field retest

* **The earlier "translation barely realized" hypothesis was a distant-scene (GC007) artefact.** On 13 scenes
  (`reproduction/near-field-motion.md`, stock checkpoint, seed 42, scripted paths, monocular odometry): forward expands in every
  scene (median 1.78x, range 1.07 to 4.79), backward contracts in every scene (median 0.49x), and turning realizes a mean
  176.6 degrees of 180. Camera control is not the general bottleneck. The gate-failure causes for GC017, GC015 and GC027
  remain untested (player, judge or task objective); `reproduction/score-drivers.md` carries the same correction. The
  "turn rate about 0.9x" and "round trip never returns" notes above are scene-specific for the same reason.
* **WARNING, krasnaya-zvesda weights volume.** The `worldsync-lingbot-weights` volume in that Modal workspace was filled from the
  Hugging Face repo head by `scripts/modal_seed_weights.py`. It is not hash-checked against the pinned checkpoint manifest
  (`reproduction/lingbot-weights.json`), so it may differ from the pinned revision the nissan-pow volume holds. Do not use it
  for scored or reproducibility runs until its files are verified against the manifest. Also: the krasnaya-zvesda and
  seraphina-ink-88 workspaces could not launch GPU functions when tried (Modal asked for a payment method).

## Powered judged comparison and Gemini scripts (added later)

* `reproduction/powered-comparison.md` is **blocked, no scores exist**: the krasnaya-zvesda and seraphina-ink-88 workspaces
  refuse every GPU type (payment method required), and the nissan-pow workspace was reserved for other forks. Its analysis
  and manifest scripts (`scripts/analyze_powered.py`, `build_powered_manifest.py`, `analyze_gemini_scores.py`,
  `build_gemini_manifest.py`, `inspect_gemini_episode.py`) and `scripts/modal_gpu_probe.py` are on main. The episode runner
  `scripts/modal_lingbot_powered.py` and its rollout flags stay on branch `worktree-agent-a5f22f4098cb528aa`.
* Two Gemini judge adapters now exist on main: `worldsync/judge_gemini.py` (main's own transport shim used by `scripts/rejudge.py`)
  and `worldsync/gemini_vertex_judge.py` (runs PlayWorld's released scorer unchanged over Vertex). They overlap; pick one for
  scored runs and state which in the report. The Gemini rescoring report of branch `worktree-agent-a0a0580ab4da8a3e8`
  (`reproduction/gemini-rescore.md`) was not committed when this was written, so it is not on main; its interim finding was a
  weak per-episode agreement with the Opus judge (Spearman about 0.32 over 44 episodes) and a much more lenient Gemini score on one episode.

## Persistent runner merge (added later)

The persistent Modal runner is merged: `scripts/modal_lingbot_runner.py`, `scripts/lingbot_runner_client.py`, `worldsync/lingbot_runtime.py`,
`worldsync/lingbot_runner_core.py` and their helper scripts and unit tests. The rollout change is small and default-neutral:
`lingbot_rollout.main(argv=None, runtime=None)` now parses `argv`, returns the report, builds the pipeline through
`build_pipeline` (or `runtime.pipeline_for` for a warm runtime), resets the CUDA peak-memory counter per job, avoids
re-inserting the LingBot source on `sys.path`, and gains `--lora PATH` (only with `--controlled-ablation`, native fixed controls
and fixed actions). One-shot `python -m worldsync.lingbot_rollout` behaves as before.

Validation on the runner branch (older rollout base): 3.6x faster per job (49.6 s warm vs 180.7 s one-shot on an H200), and
pixel hashes identical to one-shot for stock, 46/6, budget memory and the LoRA on the same GPU model, with no state leakage
across 7 jobs (`reproduction/runner-validation/`). **Not re-run on current main:** the merged rollout has only had unit tests
(137 pass) and a `--dry-run` through `main(argv)`. Re-run `scripts/compare_runner_results.py` on a GPU before relying on
bit-exactness. Bit-exactness holds only within one GPU model (Modal can serve an H100 request on an H200).
The runner needs a Modal workspace that can launch GPUs; the two newer workspaces could not when tried.

