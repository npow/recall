# Pre-registered pass criterion for "beat Genie 3 (2.12)"

Written before any four-dimension number exists, so the target cannot move afterwards.

**Goal.** A PlayWorld-style overall score above **2.12** (Genie 3, published) using the paper's formula:
overall = equal mean of the four dimension means (GC, IF, insight, out-of-sight).
Published references: LingBot-World2 (hosted) 1.82 (GC 2.04, IF 2.13, insight 1.95, OE 1.16); Genie 3 2.12.

**What counts.**
- Judge: the released PlayWorld VQA scorer, rubric, instruction gate and aggregation, via Gemini
  (Vertex, `gemini-3.1-pro-preview`; the paper used Gemini 3.1 Pro, so this is a flagged deviation).
- Baseline: stock 18/6 run under the *identical* harness (same player, end-guard, retries, cases, seeds,
  judge). Only the delta over this baseline is credited to the method.
- Config: frozen before the final run. The arm is chosen from development cases and seeds, not from the
  run that is reported.
- Final claim: all 171 cases (or the untouched held-out split reported separately), never a
  48-case subset. Bootstrap CIs over cases. The claim holds only if the lower CI bound of
  (method overall) is above 2.12 **or**, if we report only a point estimate, it is labelled as such.
- A 48-case, one-seed run is a **pilot**: it can motivate the final run but cannot establish ">2.12".

**Arithmetic.** Beating 2.12 needs the four dimension means to total more than 8.48.
Local Gemini stock so far: GC about 2.4, OE about 1.2; IF and insight not yet run locally.
LingBot's published dimension means (2.04 + 2.13 + 1.95 + 1.16 = 7.28) average 1.82, so matching 2.12 means gaining about
1.2 in total across the four dimension means (+0.30 overall). Our best judged gain so far is about +0.40 on a two-dimension
(GC and out-of-sight) subset with a confidence interval that includes zero, and it is unclear it transfers to IF and insight.
Expect the honest answer to be "not established".

**If the measured ceiling is below 2.12 we say so.** The criterion is not reduced to clear it.

**Known confounds to report with any number.** Local checkpoint vs hosted LingBot; our player and
end-guard vs the paper's player; Gemini preview model name; case subset; seed count; end-guard and
retries are our additions and also lift stock.
