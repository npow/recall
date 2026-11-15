# Corrected-controller validation: canonical controller vs legacy player (12 cases)

**Scope: validation on a 12-case sample, not a benchmark score.** One seed (42), one GPU, three Gemini calls per video.

## Setup

- **Cases:** per split, task ids ordered by sha256('42:'+task_id); first 3 supported by the canonical-controller preflight: GC GC024 GC022 GC030, IF IF012 IF006 IF022, In-sight OE066 OE009 OE020, Out-of-sight OE055 OE028 OE062.
- **Legacy arm:** frozen full-run config at `5b8a5b8` (`reproduction/full-run-config.json`): `--player-retries 2 --text-prompt both --offload-text-encoder`, 40 chunks; insight `--text-prompt caption` at 60 chunks; Haiku 4.5 player.
- **Corrected arm:** Codex's canonical controller on current `main` via `--validate-controller` (task objective as text, canonical timeline, released hold service, 60-chunk budget). Hold-service token cap raised from the released 60 to 400 (`WORLDSYNC_HOLD_MAX_TOKENS`) because Haiku's replies exceed 60 tokens and every hold check otherwise fails.
- **Hardware:** both arms on the same local RTX PRO 6000 Blackwell (shared with other projects). Not bit-comparable to the Modal frozen run.
- **Player transport:** Haiku 4.5 (`claude-haiku-4-5-20251001`) through the local CLIProxyAPI; the project API key is spend-capped.
- **Judge:** `worldsync/gemini_vertex_judge.py` (released rubric, gate and aggregation), `gemini-3.1-pro-preview` on Vertex, three independent calls per video (separate caches).

## Results

| Dimension | Legacy | Corrected | Corrected − legacy |
|---|---:|---:|---:|
| GC (n=3) | 2.53 | 2.96 | 0.42 |
| IF (n=3) | 1.33 | 1.73 | 0.39 |
| In-sight (n=3) | 1.22 | 1.20 | -0.02 |
| Out-of-sight (n=3) | 1.00 | 1.00 | 0.00 |
| **Overall (equal mean of dimensions)** | **1.52** | **1.72** | **0.20** |

Paired per-case difference (corrected − legacy): mean 0.20 over 12 pairs, case-bootstrap 95% interval [-0.68, 0.99]; 5 better / 4 equal / 3 worse.

- **Gate verdicts, Legacy:** pass 5, partial 13, fail 9, not applicable 9 (of 36 judge calls).
- **Gate verdicts, Corrected:** pass 9, partial 13, fail 5, not applicable 9 (of 36 judge calls).

## Per case

| Case | Legacy | Corrected | Δ | Legacy gates | Corrected gates | Frozen stock (Modal H200, 1 call) |
|---|---:|---:|---:|---|---|---:|
| gc:GC024 | 4.60 (3/3) | 1.00 (3/3) | -3.60 | partial, partial, partial | partial, partial, partial | 1.00 |
| gc:GC022 | 1.80 (3/3) | 4.60 (3/3) | 2.80 | partial, partial, partial | pass, pass, pass | 2.40 |
| gc:GC030 | 1.20 (3/3) | 3.27 (3/3) | 2.07 | partial, pass, pass | partial, partial, partial | 1.20 |
| if:IF012 | 1.00 (3/3) | 1.33 (3/3) | 0.33 | fail, fail, fail | pass, pass, pass | 1.00 |
| if:IF006 | 2.00 (3/3) | 1.71 (3/3) | -0.29 | partial, partial, partial | pass, pass, pass | 2.00 |
| if:IF022 | 1.00 (3/3) | 2.13 (3/3) | 1.13 | fail, fail, fail | partial, partial, partial | 1.00 |
| insight:OE066 | 1.33 (3/3) | 1.00 (3/3) | -0.33 | not_applicable, not_applicable, not_applicable | not_applicable, not_applicable, not_applicable | 1.40 |
| insight:OE009 | 1.00 (3/3) | 1.00 (3/3) | 0.00 | not_applicable, not_applicable, not_applicable | not_applicable, not_applicable, not_applicable | 3.00 |
| insight:OE020 | 1.33 (3/3) | 1.60 (3/3) | 0.27 | not_applicable, not_applicable, not_applicable | not_applicable, not_applicable, not_applicable | 2.00 |
| outsight:OE055 | 1.00 (3/3) | 1.00 (3/3) | 0.00 | fail, fail, fail | partial, partial, partial | 1.00 |
| outsight:OE028 | 1.00 (3/3) | 1.00 (3/3) | 0.00 | partial, pass, pass | partial, fail, fail | 1.00 |
| outsight:OE062 | 1.00 (3/3) | 1.00 (3/3) | 0.00 | partial, partial, pass | fail, fail, fail | 1.00 |

## Generation

| Case | Legacy (s, termination, player calls) | Corrected (s, termination, player calls) |
|---|---|---|
| gc:GC024 | 6.8125, agent_end, 8 | 59.8125, chunk_budget_exhausted, 6 |
| gc:GC022 | 6.8125, agent_end, 8 | 59.8125, chunk_budget_exhausted, 0 |
| gc:GC030 | 6.8125, agent_end, 8 | 59.8125, chunk_budget_exhausted, 0 |
| if:IF012 | 4.8125, agent_end, 6 | 59.8125, chunk_budget_exhausted, 3 |
| if:IF006 | 4.8125, agent_end, 6 | 59.8125, chunk_budget_exhausted, 3 |
| if:IF022 | 7.8125, agent_end, 9 | 59.8125, chunk_budget_exhausted, 3 |
| insight:OE066 | 59.8125, chunk_budget_exhausted, 0 | 59.8125, chunk_budget_exhausted, 0 |
| insight:OE009 | 59.8125, chunk_budget_exhausted, 0 | 59.8125, chunk_budget_exhausted, 0 |
| insight:OE020 | 59.8125, chunk_budget_exhausted, 0 | 59.8125, chunk_budget_exhausted, 0 |
| outsight:OE055 | 9.8125, agent_end, 11 | 59.8125, chunk_budget_exhausted, 6 |
| outsight:OE028 | 4.8125, agent_end, 6 | 59.8125, chunk_budget_exhausted, 2 |
| outsight:OE062 | 7.8125, agent_end, 9 | 59.8125, chunk_budget_exhausted, 0 |

## Caveats

- Twelve cases and one seed: a single episode varies by about ±1 point between seeds, so only large differences are informative.
- The arms differ in more than the controller: text (caption+objective vs objective only; insight caption vs "wait and observe") and chunk budget (40 vs 60 for moving cases). This compares the two complete configurations as specified.
- Hold-service token cap raised (above). Proxy transport for the player.
- Episode length differs sharply: the legacy player sends END once the reference plan's legs are done, so its moving-case videos are 5-10 s long; every corrected-arm video runs the full 60 s budget. Short videos leave no time for a revisit, which bears on the out-of-sight gate; long ones give the judge more to fault.
- Sanity check: the legacy arm's scores equal the frozen Modal stock run's on 7 of 9 moving cases (GC024 4.60 vs 1.00 and GC022 1.80 vs 2.40 differ), so the local legacy arm broadly reproduces the frozen configuration. Insight OE009 and OE020 score lower locally (1.00 vs 3.00, 1.33 vs 2.00).
- Missing judgements: none.
