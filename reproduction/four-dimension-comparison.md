# Four-dimension comparison: stock vs bounded pose memory (PARTIAL pilot, seed 42)

**Status: partial.** Only the IF dimension is complete (12 cases x 2 arms) plus 10 of 24 insight
episodes. No out-of-sight or GC episode finished: the one-shot run was stopped at the coordinator's request
so a faster warm-container queue can run the full set. No overall figure exists yet, so nothing here is
comparable with the paper's 1.82 or Genie 3's 2.12. Exact flags, case list and finished episode
directories: [four-dimension-config.json](four-dimension-config.json). Code: `worldsync/fourdim.py`,
`scripts/modal_lingbot_fourdim.py`, `scripts/sync_fourdim.py`, `scripts/analyze_fourdim.py`.

## Protocol

* 48 cases, 12 per dimension, chosen by a salted hash of the case id (never by outcome). GC013 is
  excluded in advance because its released image contradicts its caption and rubric
  ([input-audit-gc013.md](input-audit-gc013.md)); the next case in hash order replaces it.
* Arms differ only in the memory setting. Common flags: `--player-retries 2 --text-prompt both
  --offload-text-encoder`, seed 42, up to 40 chunks (60 for insight). Stock adds nothing; budget adds
  `--budget-memory --budget-rope native --budget-retrieved 10`.
* Player: Haiku 4.5 reference-guided (Player v6 with END challenge). Insight cases hold WAIT with no player.
* Primary judge: the released PlayWorld Gemini rubric, gate and aggregation through Vertex AI
  (`gemini-3.1-pro-preview`, gcloud login). Secondary: Opus (`worldsync.judge`). Overall = equal mean of the
  four dimension means; intervals are 95% bootstrap CIs resampling cases.
* Deviations (also in [lingbot.md](lingbot.md)): 48 of 171 cases, local 14B causal-fast checkpoint vs the
  hosted service, our player vs theirs, a preview Gemini model name, `--text-prompt both`, one seed.
  `reproduction/state.json` still says `gemini_calls_authorized: false`; Gemini scoring was authorised by the
  user for this comparison in the running session.

## Partial results (Gemini primary)

| | GC | IF | insight | out-of-sight | overall |
|---|---:|---:|---:|---:|---:|
| paper, LingBot-World2 (hosted) | 2.04 | 2.13 | 1.95 | 1.16 | 1.82 |
| paper, Genie 3 | | | | | 2.12 |
| ours stock (cases 0/12/6/0) | n/a | 2.00 | 1.17 | n/a | n/a |
| ours budget memory (cases 0/12/4/0) | n/a | 1.78 | 1.10 | n/a | n/a |

Paired budget minus stock (cases scored in both arms):

* IF: Gemini -0.22 [-1.11, +0.51], 3 wins / 6 ties / 3 losses of 12. Opus -0.29 [-0.83, +0.22] of 9.
* insight: Gemini +0.10 [0.00, +0.20] of 4 pairs; Opus -0.10 [-0.60, +0.30]. Too few pairs to read.
* Gemini gate on IF: stock 4 fail / 3 partial / 5 pass; budget 5 / 1 / 6.

Full table and JSON: [four-dimension/fourdim-table.md](four-dimension/fourdim-table.md),
[four-dimension/fourdim-analysis.json](four-dimension/fourdim-analysis.json).

## What this does and does not show

* **IF shows no memory effect, and could not.** All 24 IF episodes ended on the player's END after about
  4.8 s of video (stock 4.8 s, budget 4.9 s). The stock window covers about 4.5 s, so the cache essentially
  never evicts anything and bounded memory has nothing to retrieve. A neutral IF result is what to expect;
  it says nothing for or against revisit memory.
* Stock IF at 2.00 against the paper's 2.13 is the only dimension-level sanity check so far (12 cases,
  one seed, different player and checkpoint); it does not establish reproduction.
* Insight stock is 1.17 against the paper's 1.95 on 6 of 12 cases. Stationary 60 s WAIT episodes may be
  scored differently here (no player, `--text-prompt both` with the prompt "wait and observe"). This gap is
  unexplained and the sample is small.
* The budget arm's peak GPU memory was 81.9 GB on IF (stock 72.8 GB), so it fits an 80 GB H100 only with
  the text-encoder offload and expandable segments.

## Where this still breaks

* Not run: out-of-sight and GC (where revisits happen), the second seed, and Opus for 4 IF episodes
  without a numeric score. 9 in-flight episodes were cancelled when the one-shot app was stopped.
* Single seed; the Gemini-vs-Opus agreement is weak per episode (earlier cohorts: Pearson about 0.3), so
  conclusions that rest on one judge are fragile.
