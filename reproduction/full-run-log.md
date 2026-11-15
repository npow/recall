# Full-run log (label `full1`)

Chronological record of everything done to the full 171-case x 2-arm run after the config was frozen
(`reproduction/full-run-config.json`, committed before launch). Infrastructure changes never touch the
frozen rollout flags; the one flag amendment is declared in the config's `amendments` before any episode of the
affected dimension existed.

| Time (PST, 2026-11-14) | Event |
|---|---|
| ~22:40 | Config frozen (`a5597f3`), code commit `5b8a5b8`. |
| 22:41 | Launch: 171 cases x {stock, budget}, seed 42, 5 stock workers (H100 class) + 5 budget workers (H200). Cases interleaved across the four dimensions. |
| 22:49 | First budget episode finished (160 s). No stock worker had a GPU: workspace at its container cap (other sessions' containers). No H100 was ever scheduled. |
| 22:50 | Coordinator request: queue insight (60-chunk, heaviest) last; move stock off H100. Queues rebuilt with `scripts/requeue_full_run.py`: IF / out-of-sight / GC interleaved first, 30 insight cases last, both arms. Stock workers cancelled on H100 and re-spawned on H200 (class cap raised to 10 per class). **Side effect:** the cancelled workers had just popped 3 stock jobs (IF010, OE028, OE059); those recorded `InputCancellation` (non-transient, not retried). Not a systematic failure: no OOM, no player error. Stale failure records deleted, the 3 jobs re-queued (back of the stock queue; `modal.Queue` has no front insertion). |
| 23:01 | First stock episode completed (stock workers started producing after the class switch). |
| ~23:25 | Amendment (coordinator, before any insight episode existed): insight uses `--text-prompt caption` instead of `both`, both arms. One budget insight job (OE025) had been taken before the reorder under `both`: kept as a sensitivity datum, regenerated under `caption`. Rebuild script: `scripts/rebuild_insight_tail.py` (dry-run verified, not yet executed; waits for the coordinator's go and the insight-length decision). |
| 23:28 | Coordinator GO: insight rebuild executed (`scripts/rebuild_insight_tail.py --text-prompt caption --chunks 60`). Verified with `scripts/verify_full_run_queues.py`: no duplicates, no queued job already completed, all insight jobs `--text-prompt caption --max-chunks 60`, all non-insight jobs `both`; 10 drain workers still running (5 budget, 5 stock on H200; the 5 cancelled H100 stock calls end as `RemoteError`, as intended). Budget queue 76 -> 77 (OE025 regenerated). **Sensitivity datum problem:** my script's `modal volume cp -r` of the old-prompt OE025 episode to `adaptive/full1_sensitivity_both/` failed silently and it then removed the original from the results volume. The datum survives locally (`outputs/adaptive/full1/episodes/budget-insight-OE025-seed42/`: rollout.mp4 + run.json, and its Gemini judgement in `outputs/adaptive/full1/sensitivity_both/gemini/episodes/`), not in the volume. |
| 23:32 | **Lost stock jobs found and fixed.** Five stock jobs (GC022, GC023, GC035, IF004, IF025) were neither completed, queued, failed nor running ~40 min after the 22:50 H100 cancellation (the cancelled workers had popped them; only 3 of the 8 popped jobs left a failure record). Detected as persistent members of `in_flight_or_missing` across consecutive snapshots while the live set rotated. Stale partial attempt directories removed, jobs re-queued ahead of the insight tail (`scripts/requeue_lost.py`). Infra only; flags unchanged. Budget arm: exactly 5 in flight, nothing lost. |

## Timing facts for the amendment

Workers reach the insight tail only after the non-insight jobs of their queue. At 23:27 the queues held 100 (stock) and
49 (budget) non-insight jobs, i.e. the budget arm starts insight at about 23:55 and the stock arm at about 00:15.
The tail must therefore be rebuilt before then (or the workers paused) to guarantee no insight episode starts under `both`.

## Throughput

Measured from per-episode completion timestamps (not the per-container mean): see `reproduction/full-run-status.json`.
