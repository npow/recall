# Second baseline and evaluation audit — November15,2026

## Current handoff status

**GPU runtime smoke validated: YES, GC010 two chunks only, at repository source commit `2beab94` (also identical for the recorded sources at `66a15bd`). Full episode, live actor requests, and benchmark scores: NOT validated.** Controller handoff code is commit `e2e422a9c4a7246c9b4e054359fa10ffeecfeb2d`. CPU checks pass; Codex completed the separate Lambda two-chunk GC010 smoke and requested termination of that instance. No full171-case run is planned by Codex. Claude on odin owns the subsequent approximately12-case controller comparison; Claude on thor owns the frozen full run, optb fleet and director experiment.

Entrypoint: `worldsync.lingbot_released_baseline`. After installing dependencies and staging the pinned model/data, run from the repo root:

```bash
WORLDSYNC_HOLD_MAX_TOKENS=400 python -m worldsync.lingbot_released_baseline --split gc --task-id GC010 --seed 42 --chunks 2 --generate --validate-runtime --output outputs/canonical-smoke-GC010
python -m worldsync.baseline_runtime_validation outputs/canonical-smoke-GC010 --output outputs/canonical-smoke-GC010-validation.json
```

For one complete controller-validation episode (explicit opt-in, default guard remains):

```bash
WORLDSYNC_HOLD_MAX_TOKENS=400 python -m worldsync.lingbot_released_baseline --split gc --task-id GC010 --seed 42 --chunks 60 --generate --validate-controller --output outputs/canonical-validation-GC010-seed42
```

Change `--split` and `--task-id` for the other development cases; each output directory must be new. IF033 remains unsupported. This flag labels the run as controller validation, not published benchmark reproduction. The artifact validator above is specific to the two-chunk GC010 smoke and must not be applied to a full episode or IF001.

Live nonrotation hold checks require a usable Anthropic SDK endpoint/key serving the exact `claude-haiku-4-5-20251001` snapshot. The local CLIProxyAPI mentioned in coordination is **not automatically compatible**: it must expose Anthropic Messages semantics to this SDK, or an explicit transport adapter is needed. Merely setting an OpenAI-compatible base URL will not suffice. API errors fail the episode; do not silently substitute a different player.


The published1.82 LingBot result has **not** been reproduced. Neither full-suite arm can reach2.12 with any valid scores for its pending judgments. The baseline correction profile is deliberately blocked from GPU generation until its controller integration is validated.

## Findings

| Severity | Finding and evidence | Disposition |
| --- | --- | --- |
| Blocking | The prospective baseline still called the legacy HY scheduler. That parser adds model-specific timing, including rotation padding, which is incompatible with the current native camera rate. GC010's90-degree outward/return objective can become465degrees outward/480degrees back, leaving a15-degree offset. | Generation guard added. The shared canonical DSL and low-level released hold service are available, but the native timeline is not integrated. Existing historical runs retain their original protocol. |
| High | The draft timeline consumed the wrong response schema: `None` crashed, `ArrowLeft` corrections disappeared, and `extend_ms` was ignored. Only the first of two queued corrections executed. | Fixed, with tests against the actual released response shape and both corrections. |
| High | Cumulative duration rounding alone does not preserve return displacement. Equal5400ms legs can receive22 and21 native intervals. Unspecified angle turns also inherited rounding error. | Draft timeline preserves commanded per-phase distance and angle through fractional step magnitudes. Tests establish commanded pose closure only. |
| Blocking | Native `compute_relative_poses(..., framewise=True)` normalizes translations by the largest interval in the entire supplied prefix. Fractional steps can change that scale mid-episode, after earlier conditioning has already been committed. | **Open.** Require a declared fixed normalization policy and a test of the actual concatenated conditioning supplied to the model. A closed command matrix alone is insufficient. |
| High | Multiple angles and return clauses cannot be reliably assigned to phases by counting numbers. Example: OE047 requests90-degree yaw,40-degree pitch and a return, giving more turn phases than explicit angles. | Ambiguous explicit plans fail closed in the draft timeline. Simple bindings remain a declared text heuristic, not validated semantic grounding. |
| High | A premature release could incorrectly add a requested extension; the released policy does not extend a released hold. | Fixed: defer release until the minimum hold, without adding extension. |
| High | Successful episode reports could be reused by episode ID alone; cache identities did not cover all task/reference/label/transport inputs. A missing new input could overwrite a previous report before preserving it. | New request fingerprints include evidence, full task, label, judge/transport, released metric source and adapter source. Previous reports are archived before input loading. Duplicate episode IDs are rejected. Historical results are not silently relabeled or rejudged. |
| High | Inconsistent gate verdicts and numeric scores, Boolean scores, or invalid rubric ranges could be retained as judge results. | Reject contradictory gates and invalid numeric scores before caching. Keep released partial-admitting aggregation unchanged. |
| High | Full-suite case counts alone did not establish unique or correct task membership. Missing judgments must not become an exact official overall. | Validate exact released split membership and unique cases/arms; reject Boolean scores. Report possible1–5 arithmetic bounds and withhold exact overall until complete. |
| Medium | Raw rollout metadata still identifies the legacy controller when a callback is supplied; observations acquire an extra JPEG stage; final diagnostics assume coarse chunk actions. | **Open, covered by the generation guard.** These must be corrected as part of native controller integration. |
| Capability | Native camera-ray conditioning is not a trained avatar/gripper action head. | Unsupported character actions remain declared limitations; no invented control capability. |
| Comparability | The exact released `gemini-3.1-pro` alias returned404 with the released client. Current saved scores use the qualified preview/Vertex setup. | Still open. No silent alias fallback or claim of exact published-score equivalence. |

## Independent saved-result check

All340 valid full-suite reports were replayed against the saved video evidence and released arithmetic: **zero video-hash mismatches, unavailable videos, numeric gate contradictions, nested task mismatches, or score-replay mismatches**. See `saved-score-alignment.json`. The cache defects above are real prospective integrity risks; this audit did not find that they changed these340 saved scores.

Two of342 judgments remain unscored: one provider quota error and one provider content block. Neither is replaced with an invented score. The stock arm is bounded approximately1.765–1.798; the method arm approximately1.713–1.747. These are arithmetic bounds, not confidence intervals. The method cannot exceed stock on this saved matrix, even at the most favorable pending values.

The released partial gate admits some videos with no actual revisit. High rubric scores on those cases do not establish memory success. Preserve that official arithmetic, and inspect navigation and landmark consistency separately.

## Validation and spending

Targeted CPU suite:72passed (`test_native_reference_timeline`, `test_judge_integrity`, `test_full1_summary`, `test_gemini_vertex_judge`, `test_released_reference_player`, `test_lingbot_controls`). No new GPU experiment or judge API generation was used for this audit. The briefly provisioned Lambda H100 finished setup/weight verification but did not run a baseline; it was terminated when the controller audit blocked generation. The account instance listing subsequently confirmed it was gone. Collaborator jobs on the local GPU were left running.

Before another corrected baseline run: integrate the canonical timeline and low-level hold service, verify fixed translation normalization and actual camera embeddings, stamp correct provenance, remove the extra observation recompression, and validate representative turns/returns without inferring success from command odometry.

## Fixed-scale conditioning follow-up

`worldsync/camera_conditioning.py` now chooses a translation denominator once from the declared canonical plan. Native relative transforms are computed with prefix normalization disabled, then divided by that fixed denominator. Extensions retain their phase's displacement; early release does not change units already used by the model. If the largest planned movement is never executed, retaining the original denominator is intentional and differs from retrospectively normalizing the shortened trajectory.

The CPU regression uses the actual native camera utilities on a5400ms-per-leg rectangle. It verifies that concatenated chunk conditioning matches native full-trajectory normalization, that the six-channel camera embeddings agree, and that composing the conditioned transforms closes the rectangle. The old prefix-normalized conditioning fails the equality check. A separate early-release/extension test establishes that later commands cannot rescale committed conditioning. All16conditioning/timeline tests pass.

This validates the normalization helper, not rollout integration or visual return fidelity. The generation guard remains in place until the controller, conditioning helper, observation encoding and provenance are integrated. No GPU or judge call was used for this follow-up.

## Controller integration follow-up

The baseline player now invokes the canonical timeline and released low-level visual hold service, with a neutral first chunk and a separate40-request budget. It no longer invokes the HY scheduler. The rollout callback path consumes quarter-step camera commands and the fixed translation denominator, saves lossless PNG observations before the released JPEG stages, stamps the supplied controller profile into the raw report, and reports actual commanded poses instead of reconstructing them from coarse chunk labels. Failed runs also update the outer protocol manifest.

CPU integration tests using actual GC010 and GC002 tasks establish the intended90-degree turn/return and rectangular command closure, original phase indices for hold checks, and invariant movement scale. The controller/conditioning/timeline/released-service suite passes37tests. The full GPU rollout hook path remains unvalidated, so the generation guard is still active. These tests do not demonstrate visual revisits or a reproduced benchmark score.

The actual rollout CPU preflight now runs with hillclimbing disabled and preserves the corrected controller provenance. A two-chunk `--generate --validate-runtime --chunks 2` mode permits bounded GPU integration validation while full generation remains guarded. It cannot be used as benchmark evidence. Failure-path tests verify that the outer protocol records model-loading errors. The combined targeted suite passes40tests.

Dataset-wide canonical-plan coverage is167/171 (see `canonical-plan-coverage.json`). IF033 has no supported shared camera DSL; OE044, OE047 and OE052 have unresolved explicit-angle bindings. These four cases remain in the benchmark inventory and cannot silently be omitted from a reported full score. The CPU coverage check does not establish visual instruction following, especially for third-person character tasks.

The three multi-return objectives now have explicit phase-angle assignments in `explicit-angle-bindings.json`, fingerprinted against the exact objective and parsed actions. A changed task invalidates the binding. Tests verify the requested yaw returns and OE047's final40-degree upward pitch. Current canonical coverage is170/171; IF033 remains unsupported. The expanded targeted suite passes44tests. These are command-conditioning checks, not visual success evidence.

Saved runtime-smoke artifacts can be checked with `python -m worldsync.baseline_runtime_validation DIRECTORY --output RECEIPT.json`. Validation requires matching inner/outer source provenance, the two canonical GC010 command chunks, fixed scale, recorded commanded poses, lossless PNG observations, matching chunk video hashes, and a29-frame16fps final video. Four synthetic artifact tests cover success and rejection of missing observations, corrupt chunks, and altered commands. Synthetic fixtures are not model evidence; the live GPU smoke still needs to complete and pass this validator.

## Saved identical-video judgments

`scripts/audit_identical_judgments.py` groups saved reports by task, video SHA, requested model/provider and gate/scoring prompt hashes. In the recovered full1 reports,21groups repeat those identities;5receive different scores, with a maximum3.2-point spread (GC037:1.8versus5.0, both gates pass). Full evidence and report hashes are in `identical-video-judge-audit.json`. This is a post-hoc subset and cannot estimate whole-suite uncertainty. Historical reports omit some request and served-model details, so matching these fields does not prove byte-identical API requests. Existing scores are preserved. Candidate promotion needs repeat judging under a declared policy, without choosing favorable replicates; a single-call increase is insufficient evidence of a small memory effect.

## GPU smoke result

Evidence: `gpu-smoke-GC010/validation.json`, `protocol.json`, `episode/run.json`, video and observations. H100PCIe80GB, stock18/6, seed42, exact task prompt,2chunks/29frames at16fps, no actor API calls. Native runtime83.166seconds including loading; peak allocated71.873GB. All recorded repository-source hashes match commit`2beab94`; the full-validation flag added at`e2e422a` was not exercised.

The validator initially rejected the current checkout because the handoff entrypoint had changed while the remote run was in flight. Validation against the archived source actually staged on Lambda passed every recorded hash. `--source-root` selects such an archive without weakening hash checks. A two-chunk output does not show the return leg, long-horizon memory, or score improvement. Odin should still validate IF001/live actor transport and the complete development cases.

Visual inspection of the saved observations shows a stable neutral view followed by rightward camera motion: the sheriff building moves to the left edge and a road becomes visible. This is qualitative response evidence only. `scripts/audit_canonical_smoke_motion.py` applies the existing conservative rigid-rotation estimator to frame pairs at4-frame intervals. All3neutral pairs pass; all4turning pairs fail its rotation-fit residual threshold (about5.1–5.5px versus3px allowed). Their raw yaw estimates must not be promoted to measured camera poses. The whole-yaw estimate is withheld in `gpu-smoke-GC010/visual-motion.json`; geometry deformation or inaccurate nominal intrinsics can explain rejection. Return fidelity remains untested.

## Live actor transport preflight

Before IF001 GPU validation, test the configured Anthropic Messages endpoint with one request and no GPU:

```bash
WORLDSYNC_HOLD_MAX_TOKENS=400 python -m worldsync.released_player_probe --output outputs/canonical-player-probe.json
```

This uses the initial IF001 image and the actual pinned released hold-check service, verifies the exact served Haiku snapshot and response schema, and records success/failure separately. It costs one actor request; it is not a scored episode. Configure the endpoint/credentials outside the repo. The command has not been run by Codex against the spend-limited project key. A simulated end-to-end test now joins the canonical callback to the actual released service/parser, including null corrections; the controller/player targeted suite passes22tests.

## Hold-service output limit

Odin reported a live IF001 truncation at the released60-token cap and a successful probe at400tokens. The commands above now explicitly use `WORLDSYNC_HOLD_MAX_TOKENS=400`; this is a deviation from the released payload, not an exact protocol reproduction. Default remains60. The controller freezes the override at construction, rejects malformed/negative values before GPU loading, includes its effective cap in preflight metadata, and records released/effective caps for each request. Truncated responses still fail even with the higher cap. A failed probe at60does not prove every task response will truncate.
