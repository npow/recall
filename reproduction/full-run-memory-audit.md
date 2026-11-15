# Was budget memory actually working in the full 171-case run?

**Verdict: yes, it ran as coded, with no merge bug and no state leakage. The null on the full run is real for this design.**
The fixed-path gain (DINO 0.30 to 0.77) reproduces exactly on the full-run code, but it does not transfer to real PlayWorld episodes, for the reasons below.

Scope: read-only audit of the full run (`reproduction/full-run/`, label `full1`, code commit `5b8a5b8`), plus five fixed-path regression jobs on one H200. Run on 2026-11-15. One seed, DINO as a crude proxy, no judge involved.

## 1. Code audit: no regression

- `worldsync/lingbot_budgetmem.py` is byte-identical between the validated commit `56cb122` and the full-run commit `5b8a5b8`.
- The rollout wiring is equivalent: same constructor arguments, `memory.begin(index, camera.poses)` right after `camera.advance(...)`, the clean flag set from `t == 0`, and `memory.close()` in `finally`.
- A new memory object is built for every episode, so a warm container cannot carry one episode's archive into the next.
- `BudgetPoseMemory` overrides `begin` and `attend`, so the later changes to `PoseKVMemory` on `main` do not reach it.
- Commanded poses are one per latent frame (3 for the first action, 4 after), which matches the latent frame indices the archive uses.

## 2. GPU regression on the full-run code (origin/main at `a631b0a`)

GC007, seed 42, 68 deg/s turns, `--budget-memory --budget-rope native --budget-retrieved 10`. DINOv2-small cosine of the final frame against frame 12. Raw: `full-run-memory-audit/regression.json`, image `regression.jpg`.

| Job | Final vs frame 12 | Validated earlier | Peak GB |
| --- | ---: | ---: | ---: |
| stock, turn_hold_return | 0.299 | 0.304 | 80.2 |
| budget, turn_hold_return (first job in the container) | 0.768 | 0.768 | 92.5 |
| budget, turn_hold_return repeated after two other jobs | 0.768 | | 92.5 |
| stock, slow 360-degree turn (30 deg/s, 12 turn chunks) | 0.630 | | 80.2 |
| budget, slow 360-degree turn | 0.578 | | 92.5 |

- **The mechanism is intact:** the validated numbers reproduce to three decimals.
- **No leakage:** the repeated budget job has identical pixel hashes on all 14 chunks.
- **The slow full turn shows the design limit:** every pose is distinct, the archive ends as frames 28 to 51 only, the start-turn frames are gone, and budget is no better than stock (0.58 against 0.63).
- A sixth job (archive of 64 frames on the slow turn) was stopped at chunk 8 of 14 when Modal use was halted. It has no usable result.

## 3. What the memory did in the real episodes

Poses were rebuilt from the executed actions of all 171 budget episodes (`scripts/audit_memory_revisits.py`, raw `full-run-memory-audit/revisits.json`). A "revisit chunk" is one whose target pose is within 2 units and 30 degrees of a latent frame older than the stock window and outside the sink.

| Split | Episodes with a revisit | Revisit chunks / all chunks | Old view still archived and retrieved | Earliest copy already evicted |
| --- | ---: | ---: | ---: | ---: |
| GC | 22 of 48 | 66 / 474 | 60 | 17 |
| Out-of-sight | 14 of 43 | 26 / 281 | 26 | 1 |
| IF | 1 of 50 | 1 / 292 | 1 | 0 |
| Insight (static camera) | 30 of 30 | 1650 / 1800 | 1650 | 1560 |

- Only 92 of 755 GC and out-of-sight chunks (12%) are revisits the stock window cannot already see. In 86 of them the matching frame was retrieved, from the same episode.
- Insight is a static camera, so every frame is a "revisit" of the same pose. There the memory only swaps in other copies of the same view.

## 4. Why it made no difference

- **Returned views are no closer to the start** (`scripts/audit_return_dino.py`, raw `full-run-memory-audit/return-dino.json`). Over 54 GC and out-of-sight cases where both arms' commanded path returns to the start pose, DINO(start frame, return frame) is 0.597 for stock and 0.596 for budget (difference -0.00, 95% CI -0.03 to +0.03). On the 7 pairs with an absence of 6 chunks or more it is +0.09, with a CI that includes zero. The arms ran different player trajectories, so pairs are by case, not by path.
- **Translation returns are not undone by the model.** Most real returns are W then S. In GC007 both arms end at the mall doorway although the commanded pose is back at the start and frames 6 to 10 were retrieved (`full-run-memory-audit/return-examples.jpg`: start, farthest, return; rows GC007, OE044, OE070, stock then budget). Retrieval is indexed by commanded pose, which is wrong when the realized camera is elsewhere. The fixed-path validation was rotation-only, where commanded and realized motion agree.
- **The 24-frame archive is a FIFO on moving paths.** Eviction drops the frame whose pose is closest to another, and ties go to the oldest. With evenly spaced poses every frame ties, so the oldest goes first: in GC, 98 of the 131 chunks with a full archive hold only the most recent frames. The practical horizon is about 8 chunks (6 s), not unbounded. The validated hold path was the favourable case, because its 24 WAIT frames are duplicates and are evicted first.
- **When a pose repeats, the older copy is evicted and the newer, possibly drifted, copy is kept.** For revisit consistency the original is the one worth keeping.
- **The start view is in the sink for both arms**, so returns to the start are not something only the memory arm can see.
- Episodes are short and single-call judge noise is about 0.35 on identical videos (see `full-run-memory-fired.md`), so a small effect would not be visible anyway.

## 5. What would have to change (not tested)

- Protect the earliest copy of each view from eviction, and break ties toward evicting the middle of a run, not the start.
- Index retrieval by realized motion (visual odometry or image features), not commanded pose, so translation returns can match.
- Evaluate on paths with real, realized revisits; most PlayWorld plans do not contain one.

## Files

- `scripts/audit_memory_trace.py`, `scripts/audit_memory_revisits.py`, `scripts/audit_return_dino.py`
- `scripts/audit_fetch_regression.py`, `scripts/audit_regression_report.py`
- `reproduction/full-run-memory-audit/` (regression jobs and results, revisit and return tables, two images)
- `scripts/modal_lingbot_runner.py`: the app name can now be overridden with `LINGBOT_RUNNER_APP` (default unchanged)
