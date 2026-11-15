# Corrected-controller validation: canonical controller vs legacy player (12 cases)

**Scope: validation on a 12-case sample, not a benchmark score.** One seed (42), each pair on the same GPU, three Gemini calls per video.

## Setup

- **Cases:** Three cases per split, sha256('42:'+task_id), retained from Odin's declared plan: GC GC024 GC022 GC030, IF IF012 IF006 IF022, In-sight OE066 OE009 OE020, Out-of-sight OE055 OE028 OE062.
- **Legacy arm:** frozen full-run config at `5b8a5b8` (`reproduction/full-run-config.json`): `--player-retries 2 --text-prompt both --offload-text-encoder`, 40 chunks; insight `--text-prompt caption` at 60 chunks; Haiku 4.5 player.
- **Corrected arm:** Codex's canonical controller at `00372892bbb644964d0f15d7366739e754031352` via `--validate-controller` (task objective as text, canonical timeline, released hold service, 60-chunk budget). Hold-service token cap raised from the released 60 to 400 (`WORLDSYNC_HOLD_MAX_TOKENS`) because a live IF001 response truncated at60tokens.
- **Hardware:** Three H100SXM workers; each complete legacy/corrected pair stays on one GPU. Inspect run.json hardware fields. Not bit-comparable to the Modal frozen run.
- **Player transport:** Haiku 4.5 (`claude-haiku-4-5-20251001`) through the local CLIProxyAPI; the project API key is spend-capped.
- **Judge:** `worldsync/gemini_vertex_judge.py` (released rubric, gate and aggregation), `gemini-3.1-pro-preview` on Vertex, three independent calls per video (separate caches).

## Results

Arm means use completed cases only. Differences use matched case pairs; unequal completion can make the displayed arm means cover different cases. Overall and its interval remain withheld until all planned cases are complete.

| Dimension | Legacy | Corrected | Paired corrected − legacy |
|---|---:|---:|---:|
| GC (L=3/3, C=3/3, pairs=3/3) | 2.33 | 1.60 | -0.73 |
| IF (L=3/3, C=3/3, pairs=3/3) | 1.56 | 2.42 | 0.86 |
| In-sight (L=2/3, C=3/3, pairs=2/3) | 3.27 | 1.11 | -2.27 |
| Out-of-sight (L=3/3, C=3/3, pairs=3/3) | 1.31 | 1.00 | -0.31 |
| **Overall (equal mean of dimensions)** | **—** | **1.53** | **—** |

Paired per-case difference (corrected − legacy): mean -0.46 over 11 pairs, split-stratified paired-bootstrap 95% interval (withheld until complete) [—, —]; 3 better / 2 equal / 6 worse.

- **Gate verdicts, Legacy:** pass 13, partial 9, fail 5, not applicable 8 (of 36 judge calls).
- **Gate verdicts, Corrected:** pass 11, partial 14, fail 2, not applicable 9 (of 36 judge calls).

## Missing-score sensitivity

Allow each missing declared judgment to take any legal score from1to5. These are deterministic completion bounds, not imputed results or confidence intervals. They hold the observed scores fixed and do not capture generation or judge uncertainty.

- Legacy overall would lie in [2.1153, 2.2264].
- Corrected overall would lie in [1.5322, 1.5322].
- Corrected minus legacy would lie in [-0.6942, -0.5831].

## Per case

| Case | Legacy | Corrected | Δ | Legacy gates | Corrected gates | Frozen stock (Modal H200, 1 call) |
|---|---:|---:|---:|---|---|---:|
| gc:GC024 | 2.60 (3/3) | 1.07 (3/3) | -1.53 | partial, partial, partial | partial, partial, pass | 1.00 |
| gc:GC022 | 2.93 (3/3) | 2.40 (3/3) | -0.53 | pass, pass, pass | pass, partial, pass | 2.40 |
| gc:GC030 | 1.47 (3/3) | 1.33 (3/3) | -0.13 | pass, pass, pass | pass, partial, partial | 1.20 |
| if:IF012 | 1.00 (3/3) | 1.20 (3/3) | 0.20 | fail, fail, pass | pass, partial, fail | 1.00 |
| if:IF006 | 2.00 (3/3) | 2.56 (3/3) | 0.56 | partial, partial, partial | pass, pass, pass | 2.00 |
| if:IF022 | 1.67 (3/3) | 3.50 (3/3) | 1.83 | partial, pass, pass | partial, partial, partial | 1.00 |
| insight:OE066 | 2.20 (3/3) | 1.00 (3/3) | -1.20 | not_applicable, not_applicable, not_applicable | not_applicable, not_applicable, not_applicable | 1.40 |
| insight:OE009 | — (2/3) | 1.33 (3/3) | — | not_applicable, None, not_applicable | not_applicable, not_applicable, not_applicable | 3.00 |
| insight:OE020 | 4.33 (3/3) | 1.00 (3/3) | -3.33 | not_applicable, not_applicable, not_applicable | not_applicable, not_applicable, not_applicable | 2.00 |
| outsight:OE055 | 1.00 (3/3) | 1.00 (3/3) | 0.00 | fail, fail, fail | partial, pass, pass | 1.00 |
| outsight:OE028 | 1.00 (3/3) | 1.00 (3/3) | 0.00 | pass, pass, partial | fail, pass, partial | 1.00 |
| outsight:OE062 | 1.92 (3/3) | 1.00 (3/3) | -0.92 | pass, pass, partial | partial, partial, partial | 1.00 |

## Generation

| Case | Legacy (s, termination, player calls) | Corrected (s, termination, player calls) |
|---|---|---|
| gc:GC024 | 6.8125, agent_end, 8 | 59.8125, chunk_budget_exhausted, 6 |
| gc:GC022 | 6.8125, agent_end, 8 | 59.8125, chunk_budget_exhausted, 0 |
| gc:GC030 | 7.8125, agent_end, 9 | 59.8125, chunk_budget_exhausted, 0 |
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

- Twelve cases and one seed; this cannot establish the full benchmark result or seed robustness.
- The arms differ in more than the controller: text (caption+objective vs objective only; insight caption vs "wait and observe") and chunk budget (40 vs 60 for moving cases). This compares the two complete configurations as specified.
- Hold-service token cap raised (above). Proxy transport for the player.
- Episode length differs sharply: the legacy player sends END once the reference plan's legs are done, so its moving-case videos are 5-10 s long; every corrected-arm video runs the full 60 s budget. Short videos leave no time for a revisit, which bears on the out-of-sight gate; long ones give the judge more to fault.
- The frozen-stock column is a separate hardware/runtime run with one judgment per video, versus three here. Matching individual scores does not establish reproduction or isolate the cause of cross-run differences.
- Missing judgements: L-insight-OE009: 2 scored of 3 calls. Not imputed.
