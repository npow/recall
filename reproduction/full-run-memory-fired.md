# Did bounded pose memory act in the full run? (exploratory)

**Exploratory, post hoc.** Slices were chosen after seeing the full-run scores; nothing here is a pre-registered test.
Source: the 171-case, two-arm full run (`full1`, seed 42, stock 18/6 vs `--budget-memory --budget-rope native --budget-retrieved 10`),
Gemini single-call scores, read on 2026-11-15 about 03:00 PST while a judge-repeat study and a 7-task parser-fix rerun were still in progress.
170 of 171 cases are paired (insight OE013 was unscored in the stock arm). The official per-dimension table comes from the full-run analysis, not from here.
Tables: [full-run-memory-fired/tables.md](full-run-memory-fired/tables.md); script: `scripts/analyze_memory_fired.py`.

## Findings

1. **Memory did act in most episodes.** Retrieval replaced attention in at least one chunk in 136 of 171 budget episodes
   (GC 43/48, IF 34/50, insight 30/30, out-of-sight 29/43). All 171 ran with native RoPE as configured.
   But outside insight the episodes are short: median 8.3 s (GC), 4.8 s (IF), 5.8 s (out-of-sight) against a 4.5 s stock window,
   so retrieval fires in a median of 4, 1 and 2 chunks. 35 budget episodes never used memory at all.

2. **No gain where memory fired.** Paired budget-minus-stock delta: all 170 pairs -0.05 [-0.25, +0.14];
   fired (135 pairs) -0.15 [-0.38, +0.08]; fired in 5 or more chunks (68 pairs) +0.05 [-0.30, +0.40].
   Out-of-sight where memory fired is negative: -0.30 [-0.60, -0.04] (29 pairs, 1 win, 21 ties, 7 losses); uncorrected, one of many slices.

3. **The "never fired" slice is a placebo, and it shows +0.30.** In the 35 pairs where retrieval never fired, the budget arm
   scores +0.30 [+0.02, +0.65] higher although the method did nothing. In 28 of those pairs every chunk's pixel hash is identical in
   both arms (the same video content), yet the judge scored 5 of them differently, by up to 4.0 points
   (OE046 1.0 vs 5.0, IF043 2.0 vs 5.0, IF021 1.0 vs 3.0, IF050 2.33 vs 3.0, GC033 4.2 vs 4.4); mean absolute gap 0.35.
   So a single Gemini call per video carries noise of the same size as every method effect we have reported.
   The other 7 not-fired pairs differ because the player took different actions.

4. **The earlier +0.32 does not replicate on its own cases.** The 14 cases of the earlier powered cohort score, inside this run,
   stock 1.59 vs budget 1.62: +0.04 [-0.52, +0.73] (4 wins, 6 ties, 4 losses). GC +0.54 [-0.29, +1.74] (driven by GC027 1.2 -> 5.0),
   out-of-sight -0.46 [-1.04, -0.04]. The 7 long-plan cases give -0.20, the 7 original cases +0.28.
   The earlier cohort also used `--end-guard` and caption-only text, which this run does not, so harness and noise are confounded;
   given finding 3, noise alone is a sufficient explanation and case selection is not needed to explain it.

5. **Half the benchmark sits at the floor.** 86 stock and 89 budget episodes of 170 score exactly 1.0. Gate verdicts are near identical
   across arms (stock 32 fail / 25 partial / 81 pass, budget 27 / 27 / 84; 32 insight not applicable).

## What it means

- The full-run null is not because memory was switched off or misconfigured; it ran, in most episodes, and did not help under this harness.
- Reference plans are short, so memory has little to recall outside insight; in insight it fires in 56 of 60 chunks and the delta is +0.06 [-0.50, +0.58].
- Judge noise on identical videos (finding 3) means differences under roughly 0.3 to 0.4 per slice are not detectable with one call per video.
  Repeated judging (in progress in the full-run fork) is needed before any small effect is claimed in either direction.

## Limits

One seed, one judge call per video, one judge model. Slices are post hoc and uncorrected. "Fired" is from the rollout's own trace
(`memory.trace[*].attention_replaced`), not an independent measurement. Arms did not always take the same player actions
(100 of 170 pairs have identical episode length), so fired vs not-fired also differs in episode length and dimension mix.
