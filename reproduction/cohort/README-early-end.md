# Why adaptive episodes end early (diagnosis)

Source: saved `run.json` player decisions of the first cohort (`cohort-stock-*`, seed 42) and the
reference plans in `external/PlayWorld/data/*/data.json`. Per-episode listing:
`python3 scripts/summarize_episodes.py "outputs/cohort0/cohort-stock-*"`.

1. **Mostly not a player bug: the plans are short.** The human reference plans hold actions for
   only 1.8 to 18 s in total (development split, `scripts/plan_durations.py`). OE057 is
   `hold(Left,1800ms)` then `hold(Right,1800ms)`, so a faithful episode lasts 3.6 s. The 40-interaction
   budget is a cap, not the typical length. The player followed the plan and ended with it.
2. **Short plans never exceed the stock cache.** The stock window is 18 latent frames (about 4.5 s)
   and the first frames stay as sink. Most tasks absent the start view for 2 to 4 s, so a larger cache
   has nothing extra to recall. This explains the 46/6 null result on the first cohort.
3. **False completion claims exist.** GC007: the plan is `hold(W,7200ms)`, `hold(S,7200ms)` (no turn),
   but the player walked back, turned 180 degrees, and claimed success with intended heading 180
   and position 3 units from the start. GC027 and OE026 ended with intended heading far from the plan's.
   The prompt says END only when the goal is visually achieved, and the generated view is plausible
   even when it is not the start view (scene identity drift), so the visual check cannot catch this.
4. **Tasks that do test long absence exist.** Development cases with plans of 10 s or more:
   GC005 (18 s), GC007 (14.4 s), OE044 (12.8 s), GC015, OE056, OE055, GC025, GC019. They were
   chosen by plan duration only, before seeing any outcome.

Fix: `--end-guard` (see `reproduction/lingbot.md`), plus a long-horizon cohort of the cases above.
