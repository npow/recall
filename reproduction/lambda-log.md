
## Codex baseline validation leases

- `6d885894eff4432f994edf8830ced221`: gpu_1x_h100_pcie, launched2026-11-15T10:45:04.794574+00:00, maximum60minutes, $3.29/hour, statusterminated. Purpose:Corrected stock baseline validation; no algorithm hillclimbing. Termination request:2026-11-15T11:09:23.346658+00:00.
- `05682393d6104a6aba07e830cc1bca31`: gpu_1x_h100_pcie, launched2026-11-15T11:20:44.274756+00:00, maximum60minutes, $3.29/hour, statusbooting. Purpose:Corrected stock baseline validation; no algorithm hillclimbing. Termination request:pending; automatic deadline active.
# Lambda instance log

| Session | Instance | Type | Launched (UTC) | Terminated (UTC) | Purpose |
|---|---|---|---|---|---|
| Claude odin 93395c | 8 x worldsync-b (ba679e45, 0d749c31, 86c1a39a, d3a65dad, d252a5ab, 9f2f16b0, 14e45fe4, 538388b7) | 5 x gpu_1x_h100_sxm5, 3 x gpu_1x_h100_pcie | 2026-11-15 10:45 | 10:50 (14e45fe4 stuck in terminating until about 11:05) | Launched before the orchestrator was ready; terminated unused. About $5. |
| Claude odin 93395c | 483ea84f worldsync-test | gpu_1x_a10 | 2026-11-15 10:51 | 10:55 | Setup-only test of scripts/lambda_run.py (env, weights, dry-run). |

- Codex termination requested for `05682393d6104a6aba07e830cc1bca31` at 2026-11-15T11:32:09.149783+00:00; two-chunk smoke complete and artifacts collected locally. No full-episode or171-case follow-up launched.

- Codex verified `05682393d6104a6aba07e830cc1bca31` absent from the live account instance listing at 2026-11-15T11:36:58.620027+00:00. All Codex-owned baseline leases are closed.

---

<!-- Entries below are from the option-B fleet session (branch worktree-agent-aa584524eecbcf42f); entries above are from the other sessions. -->

# Lambda Cloud log (option B infrastructure)

Hard spend cap for this work: **$150**. No secrets are recorded here. Times are PST on 2026-11-15.

## Prices and availability (API query at 03:00)

| Instance type | $/h | GPU memory | Capacity at query time |
|---|---:|---|---|
| gpu_1x_h100_pcie | 3.29 | 80 GB | us-west-3 |
| gpu_1x_h100_sxm5 | 4.29 | 80 GB | us-south-2 |
| gpu_2x_h100_sxm5 | 8.38 | 2 x 80 GB | us-southeast-1, us-south-3 |
| gpu_1x_a100_sxm4 | 1.99 | 40 GB (too small) | us-east-1, us-west-2 |
| gpu_8x_a100 | 15.92 | 8 x 40 GB (too small) | several |
| gpu_1x_gh200, gpu_1x_b200_sxm6, gpu_4x/8x_h100, gpu_8x_a100_80gb | - | - | none |

Stock-sized episodes need about 72 GB with `--offload-text-encoder` and the expandable-segments allocator,
so only the H100 types fit. The cheapest is `gpu_1x_h100_pcie`.

## Instances

All are `gpu_1x_h100_pcie` (NVIDIA H100 PCIe, 79.2 GiB), us-west-3, $3.29/h, SSH key `thor`.

| Instance id | Name | IP | Launched | Terminated |
|---|---|---|---|---|
| 672e2eee53114b13a278c1282e3c9dc5 | worldsync-optb-1 | (ip redacted) | 03:00 | (running) |
| 8709a102261c40d9be940070a39d5454 | worldsync-optb-2 | (ip redacted) | 03:19 | (running) |
| 6f22d31d20d243e2b9b974f8654b9116 | worldsync-optb-3 | (ip redacted) | 03:19 | (running) |
| 785446b21d8c447283f7c4dda9a87c0a | worldsync-optb-4 | (ip redacted) | 03:21 | (running) |

Instances named `worldsync-baseline-*` in the same account belong to another session and are not part of this
work or its cap (one was active at 03:00, a new one was booting at 03:21).

## Job batches and code commits

Instances never track `origin/main`. Each batch runs from an rsynced worktree at a recorded commit
(`~/<tree>/COMMIT` on the instance).

| Batch | Code tree on instance | Commit | Started | Notes |
|---|---|---|---|---|
| environment, weights, canary | `~/work` | a631b0a + the uncommitted Lambda scripts | 03:10 | weights: 17 files verified against `reproduction/lingbot-weights.json`, revision 5c33dd40 |
| director dev jobs (113 fixed-action jobs, label `director-lambda`) | `~/director` | abf1693 (branch `director`) | 03:24 (shard 0); shards 1-3 as instances come up | 4 shards by line index % 4; no player, no tunnel; results synced to the director worktree's `outputs/director-lambda/<id>/` |

| FINAL RUN `final1` (294 jobs: 171 stock + 123 director; `reproduction/final-run-jobs.jsonl`) | `~/final` | **abfa73a** (branch `director`) | 04:49 on optb-1..4; optb-5..8 as they come up | Pushed from the director worktree when its HEAD was 34d1a2f (one later commit that only adds `scripts/final_run_score.py`); `worldsync/` on every instance was verified byte-identical to commit abfa73a with a checksum rsync dry run, as were the job list and config. Jobs dealt out in file order to the least-loaded instance (8 lists, about 2.2 h each). Player through the reverse tunnel to the local proxy; 1 + 2 attempts per job, failed attempts kept; results synced every 45 s to the director worktree's `outputs/final1/<id>/`. |

The director dev batch was stopped at 04:46 on the coordinator's instruction with 68 of its jobs synced.

Instances added for the final run (same type, region, price and key), launched 04:46:

| Instance id | Name | IP |
|---|---|---|
| 24830a592c7346968f58ebdcfe174795 | worldsync-optb-5 | (ip redacted) |
| 657e4751de014e86adda202dd9e13ad0 | worldsync-optb-6 | (ip redacted) |
| 1c896c74f4014e6a81042ed096799ff5 | worldsync-optb-7 | (ip redacted) |
| 66b421e0193444548fe38e71f549ca40 | worldsync-optb-8 | (ip redacted) |

Final-run cost estimate made before launch: 294 jobs = about 17.8 GPU-hours (about $59) plus idle and setup on
8 instances for about 2.5 h (about $66 to $75), on top of about $21 spent by 04:48. Projected total about
$95 to $105, under the $140 stop line.

## Cap and teardown changes (coordinator instructions)

| Time | Change |
|---|---|
| 03:10 | Cap $150. |
| 06:32 | Standing order: terminate each `worldsync-optb-*` instance once idle, nothing left to take, and results verified synced; keep one for reruns. |
| 07:14 | Keep three instead of one (a later experiment needs them); hard stop 10:30. |
| 07:20 | **Cap raised to $300** (stop launching and report if the projection exceeds $280). Keep five instances; hard stop moved to 12:00; terminate idle instances when no job list is pending for 20 minutes. |

Terminations by the teardown loop (each after its completed jobs were verified present locally with `run.json`
and `rollout.mp4`): optb-6 07:10, optb-3 07:13, optb-8 07:17, optb-5 07:17, optb-2 07:17.
Replacements launched 07:20 for the five-instance standby: optb-9 `7a1b16e410834459b76c6367de7be595`
((ip redacted)) and optb-10 `8d92f0c9f6c54f159f82271a3a617907` ((ip redacted)).
Free disk on the instances: 879 to 880 GB of a 1 TB root volume (no separate data volume).

## Parity check (fixed 8-chunk stock canary, GC007 `turn_return`, seed 42, text encoder offloaded)

| Where | GPU | Chunk 0 pixel hash | Peak allocated | Canary time |
|---|---|---|---:|---:|
| Modal one-shot (`reproduction/runner-validation/summary.json`) | NVIDIA H100 80GB HBM3 | 19d26ac10080... | 71.8 GB | - |
| Modal warm | NVIDIA H200 | 39f5dd2072a4... | 68.7 GB | 46.9 s |
| Lambda `worldsync-optb-1` warm | NVIDIA H100 PCIe | a6bf3b0ac855... | 68.7 GB | 73.4 s |

The canary completes with the same peak memory, but the pixel hashes differ from both Modal GPU models, as they
already differ between Modal's H100 and H200. Results from Lambda are therefore **not bit-identical** to the Modal
full run; any Lambda arm must be compared with a stock arm generated on the same Lambda GPU model, or the
difference stated. Lambda's H100 PCIe is also slower per chunk (about 9 s vs about 6 s on Modal's H200).

## Adaptive validation through the player proxy (03:41 to 03:48, instance `worldsync-optb-3`, code tree `~/work` at a631b0a)

Player calls go from the instance through an SSH reverse tunnel to the Anthropic-compatible proxy on the local
machine (`ANTHROPIC_BASE_URL=http://(ip redacted):8317`, model `claude-haiku-4-5-20251001`); the project's own
Anthropic key is not used or sent. Two stock episodes with the full run's frozen flags, seed 42:

| Episode | Lambda result | Modal full-run result | Lambda time |
|---|---|---|---:|
| GC027 | completed, 9 chunks, RIGHT x9 then END | completed, 9 chunks, RIGHT x9 then END | 172 s |
| IF046 | attempt 0 failed (`PlayerResponseError: invalid player action` after 4 chunks), attempt 1 completed, 5 chunks, W x5 then END | completed, 5 chunks, W x5 then END | 95 s (185 s with the failed attempt) |

Same player decisions and episode lengths as the Modal run; pixel hashes differ (different GPU model, as in the
canary). Peak allocated 72.6 GB. One player response was invalid and was retried by the queue.

Throughput on H100 PCIe: about 10 s per chunk for stationary 60-chunk jobs (600 s) and about 15 to 19 s per
chunk including the player call for adaptive episodes. Projection for one 171-episode arm:
141 player-driven episodes x about 180 s + 30 insight episodes x about 600 s = about 12 GPU-hours,
about **$40 per arm** at $3.29/h (about 3 h on 4 instances, 1.5 h on 8), about $0.23 per episode.

## Spend

| Time | Instances up | Cumulative spend (estimate from launch times) |
|---|---:|---:|
| 03:49 | 4 | about $7.5 |

## Cost estimate before launching the fleet

Measured on Modal: a warm stock episode takes about 123 s (player-driven splits) and about 6.5 min for a
60-chunk insight episode. For 171 episodes that is about 141 x 2.05 min + 30 x 6.5 min = about 8 GPU-hours,
plus about 10 minutes of setup per instance (weights download and verification, model load).
At $3.29/h: about **$27 to $35 for one 171-episode arm**, whatever the number of instances
(4 instances finish in about 2.2 h, 6 in about 1.5 h). Two arms: about $55 to $70. Within the cap.
