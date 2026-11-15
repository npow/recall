# What drives the judged scores down (stock and 46/6)

Read-only analysis. No new generation. **Local Opus 5.5 judge with the pinned PlayWorld
rubric, so none of these numbers are comparable with the published 1.82 / 2.12.**

Data used:

* All 60 episodes of the guarded rerun (`reproduction/cohort/guarded-results.json`, branch
  `worktree-agent-a09db1298fc13e5e1`): 15 development cases x {stock 18/6, cache 46/6} x
  seeds {42, 43}. 54 scored, 6 failed (3 generation, 3 judge).
* The 28 seed-42 `judge.json` files (gate verdict, per-category rubric scores and evidence),
  pulled from Modal volume `worldsync-lingbot-results`. Seed-43 category scores were not pulled;
  seed-43 gate verdicts come from `guarded-results.json`.
* The judge's Chinese gate and rubric evidence was read and paraphrased by me.
* The first cohort (`worktree-agent-a5112f101863ed664`) and the GC007 v3 episode were not
  re-analysed. The realized-motion numbers are from `reproduction/realized-motion/` (branch
  `worktree-agent-a1110f346db13fb8f`).

## How the score is formed

`instruction_gate.py` first decides `pass` / `partial` / `fail` on whether the requested
trajectory was executed. A `fail` forces the final score to **1.0** and the rubric is never
applied (`rubric` is null). Otherwise visual-consistency category scores give 1-5.

| Gate verdict | Episodes (of 54 scored) | Mean score |
| --- | --- | --- |
| fail (forced 1.0) | 6 (all GC) | 1.00 |
| partial | 24 (16 GC, 8 OE) | 2.43 |
| pass | 24 (7 GC, 17 OE) | 2.29 |

By dimension: **GC mean 2.56 (n=29), OE mean 1.80 (n=25)**. Within GC the gate matters
hugely (fail 1.0, partial 2.68, pass 3.63). Within OE the gate barely matters (partial 1.94,
pass 1.74), which already says OE is limited by something else.

## Ranked failure causes

| # | Cause | Count | Examples | Evidence |
| --- | --- | --- | --- | --- |
| 1 | **The world does not execute the commanded camera trajectory** (gate fail or partial on trajectory progress) | 30 of 54 scored (56%): 6 fail, 24 partial. In GC it is 22 of 29 (76%) | GC007 (no approach to the entrance, scale unchanged, "near continuous backing up"), GC005 (no 360° orbit, camera drifts to sky or base), GC027 (360° turn never completes), GC017 / GC015 (turn under 180°, return leg not recognisable), GC024 (backward walk `S` replaced by a viewpoint jump or forward walking), OE056 (overshoots the start view) | Gate text per episode. Realized-motion fork: realized yaw is 0.89-0.92 of commanded, forward approach is only x1.04 over 4 unit steps, and scaling translation x3 made turning worse. 46/6 leaves the verdict counts unchanged (3 fail each arm). |
| 2 | **Hidden-state / event evolution is weak in out-of-sight tasks** | 25 of 25 OE episodes score <= 3.0, mean 1.80; 17 pass the gate and observe the revisit but still score 1.74 | OE057 (cream barely changes, no cherries, piping bag turns into a rigid tube), OE056 (red and blue stay frozen, dropper becomes a glass rod), OE026 | Category means (seed 42): hidden_state_progression 1.90 (n=10), process_accumulation 2.00 (10), revealed_state_difference 2.10 (10), subject_identity 1.58 (12). 46/6 does not help: OE with revisit observed 1.75 stock vs 1.73 46/6. |
| 3 | **Appearance / identity drift on revisit** (GC) | Not counted per episode (only GC019 read in detail). GC category means: identity_id 2.67, identity_texture 2.83, relative_position 3.04, color 3.75 (seed 42, n=12) | GC019 stock: rusty locomotive returns as a shiny new one, text turns to gibberish, striped pole becomes a green pole | This is the memory failure. GC episodes with revisit observed: stock 3.06 (n=5), 46/6 3.50 (n=4). Suggestive, not established. |
| 4 | Infrastructure failures | 6 of 60 (10%): 3 generation, 3 judge | GC017 46/6 both seeds (judge), OE056 stock s42, OE069 46/6 s42, GC019 46/6 s43 (generation) | Leaves arms unbalanced, so use paired seeds only. |
| 5 | Player behaviour | Early END is largely handled by `--end-guard`; guard fired on GC005, GC007, GC024 | GC007 and GC005 still fail the gate after the guard, so the world, not the END decision, is the limit | `reproduction/cohort/README-early-end.md` |

I did not read the gate text of the 8 OE partials (only OE056, overshoot), so their split between cause 1 and cause 2 is unchecked.

## Headroom

* **GC:** lifting the 6 forced-1.0 episodes to the GC partial mean (2.68) is about +0.35 on GC.
  Lifting all GC fail and partial episodes to the GC pass mean (3.63) would give about +1.07,
  but the pass episodes (for example GC025) may simply be easier tasks, so treat that as a ceiling.
* **OE:** the lowest dimension in both the paper (1.16 for LingBot) and our runs. It is not gate-limited
  and not memory-limited here, so its headroom is dynamics (cause 2). Unverified guess: attending to more
  history may reinforce a frozen earlier state.
* Only GC and OE were run. IF and insight were not in this cohort.

## Recommendation: control and dynamics realization

The single highest-leverage change is **making the trajectory actually happen and letting the
player see that it did**, ahead of more memory work:

1. The player's state is the *commanded* pose ("intended camera commands only; not measured visual
   motion"), so the player's Keep / Extend / Correct decisions cannot react to a world that moved less
   than commanded. Feed it **measured** motion from visual odometry on the generated frames (realized yaw,
   approach scale), and let it Extend or Correct until the realized trajectory meets the plan (yaw within
   about 10 degrees, a visible approach).
2. Recalibrate the turn rate to about 68 deg/s so a commanded 180 degrees is realized (realized/commanded
   is 0.89-0.92 today).
3. Re-test the `S` backward action: the judge wants a backward walk in place and gets a camera retreat or a
   viewpoint jump (GC024).

Why this first: cause 1 covers 56% of scored episodes and every forced-1.0 episode, GC is the only
dimension where the gate swings scores by 2.6 points, and memory (46/6) leaves the verdict counts unchanged.
Memory is the second lever, for cause 3 (GC identity drift on revisit), and it only pays once the camera
actually returns.

Caveats: 54 scored episodes, one judge, two seeds with up to about 2 points of seed-to-seed variance
(GC015, GC005), a judge that gates strictly (`partial` vs `pass` is a judgment call), and no check that
more Extend steps raise translation approach beyond the measured x1.04. The "closed loop will fix it" step
is a hypothesis to test, not a result.
