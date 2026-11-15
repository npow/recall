# Task-text confirmation

All 16 declared adaptive episodes completed. Four development cases were repeated with seed 43, and four new cases were selected by a fixed hash before generation. [Frozen plan](plan.json), [complete generation outcomes](results.json), and [validated judgment summary](summary.json).

## Complete new-seed partition

| Mode | Cases | Gemini mean |
|---|---:|---:|
| Caption |4/4|1.7500|
| Caption + objective |4/4|1.7075|

The development gain did not reproduce under the frozen confirmation rule. This point difference is too small to establish a statistical regression; it is sufficient to fail the predeclared nonnegative-new-seed condition. The change is **not promoted**. Per-case results include both improvements and a drop from 4 to 2 on IF042; do not interpret the mean as a universal effect.

## Held-out cases: one judgment pending

All 8 held-out episodes exist, but only 7 judgments are available. Vertex returned 429 RESOURCE_EXHAUSTED for IF024 with combined text. No held-out or combined-cohort mean is reported; a scorer transport failure is not assigned a model-failure score. [Earlier scoring failures](failed-scoring/) are retained.

The initial gcloud default account/project became unusable. Judging was scoped to the configured personal login and an existing billed research project after enabling Vertex. Results record the project and served model version. Model, prompts, sampling and aggregation remain the same `gemini-3.1-pro-preview` protocol.

## Recovery and integrity

The local generation driver exited 143 before writing its result bundle. Cloud execution completed all 16 jobs, each with exactly one committed attempt. The [recovery script](../../scripts/recover_text_confirmation.py) recovered every original run without regeneration or successful-attempt selection. Client timing and returned base-weight fingerprints were unavailable and are not invented. Source, adapter, player, checkpoint and backend hashes agree across all 16 runs; image hashes and seeds match the frozen plan.

Canonical evaluation validates every declared job, native cache/sampler settings, task/image hashes and video hashes. Use `scripts/evaluate_task_text_pilot.py --directory reproduction/task-text-confirmation --cached-only` for a local replay without judge calls. Complete partitions may be reported independently; the full cohort remains incomplete until the final judgment succeeds.
