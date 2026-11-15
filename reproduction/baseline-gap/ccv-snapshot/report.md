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
| GC (L=2/3, C=2/3, pairs=2/3) | 2.03 | 1.20 | -0.83 |
| IF (L=2/3, C=2/3, pairs=2/3) | 1.33 | 2.35 | 1.01 |
| In-sight (L=0/3, C=0/3, pairs=0/3) | — | — | — |
| Out-of-sight (L=0/3, C=0/3, pairs=0/3) | — | — | — |
| **Overall (equal mean of dimensions)** | **—** | **—** | **—** |

Paired per-case difference (corrected − legacy): mean 0.09 over 4 pairs, split-stratified paired-bootstrap 95% interval (withheld until complete) [—, —]; 2 better / 0 equal / 2 worse.

- **Gate verdicts, Legacy:** pass 6, partial 4, fail 2, not applicable 0 (of 12 judge calls).
- **Gate verdicts, Corrected:** pass 3, partial 8, fail 1, not applicable 0 (of 12 judge calls).

## Per case

| Case | Legacy | Corrected | Δ | Legacy gates | Corrected gates | Frozen stock (Modal H200, 1 call) |
|---|---:|---:|---:|---|---|---:|
| gc:GC024 | 2.60 (3/3) | 1.07 (3/3) | -1.53 | partial, partial, partial | partial, partial, pass | 1.00 |
| gc:GC022 | — (0/0) | — (0/0) | — |  |  | 2.40 |
| gc:GC030 | 1.47 (3/3) | 1.33 (3/3) | -0.13 | pass, pass, pass | pass, partial, partial | 1.20 |
| if:IF012 | 1.00 (3/3) | 1.20 (3/3) | 0.20 | fail, fail, pass | pass, partial, fail | 1.00 |
| if:IF006 | — (0/0) | — (0/0) | — |  |  | 2.00 |
| if:IF022 | 1.67 (3/3) | 3.50 (3/3) | 1.83 | partial, pass, pass | partial, partial, partial | 1.00 |
| insight:OE066 | — (0/0) | — (0/0) | — |  |  | 1.40 |
| insight:OE009 | — (0/0) | — (0/0) | — |  |  | 3.00 |
| insight:OE020 | — (0/0) | — (0/0) | — |  |  | 2.00 |
| outsight:OE055 | — (0/0) | — (0/0) | — |  |  | 1.00 |
| outsight:OE028 | — (0/0) | — (0/0) | — |  |  | 1.00 |
| outsight:OE062 | — (0/0) | — (0/0) | — |  |  | 1.00 |

## Generation

| Case | Legacy (s, termination, player calls) | Corrected (s, termination, player calls) |
|---|---|---|
| gc:GC024 | 6.8125, agent_end, 8 | 59.8125, chunk_budget_exhausted, 6 |
| gc:GC022 | None, None, None | None, None, None |
| gc:GC030 | 7.8125, agent_end, 9 | 59.8125, chunk_budget_exhausted, 0 |
| if:IF012 | 4.8125, agent_end, 6 | 59.8125, chunk_budget_exhausted, 3 |
| if:IF006 | None, None, None | None, None, None |
| if:IF022 | 7.8125, agent_end, 9 | 59.8125, chunk_budget_exhausted, 3 |
| insight:OE066 | None, None, None | 59.8125, chunk_budget_exhausted, 0 |
| insight:OE009 | None, None, None | None, None, None |
| insight:OE020 | None, None, None | None, None, None |
| outsight:OE055 | None, None, None | None, None, None |
| outsight:OE028 | None, None, None | None, None, None |
| outsight:OE062 | None, None, None | None, None, None |

## Caveats

- Twelve cases and one seed; this cannot establish the full benchmark result or seed robustness.
- The arms differ in more than the controller: text (caption+objective vs objective only; insight caption vs "wait and observe") and chunk budget (40 vs 60 for moving cases). This compares the two complete configurations as specified.
- Hold-service token cap raised (above). Proxy transport for the player.
- Missing judgements: L-gc-GC022: 0 scored of 0 calls; C-gc-GC022: 0 scored of 0 calls; L-if-IF006: 0 scored of 0 calls; C-if-IF006: 0 scored of 0 calls; L-insight-OE066: 0 scored of 0 calls; C-insight-OE066: 0 scored of 0 calls; L-insight-OE009: 0 scored of 0 calls; C-insight-OE009: 0 scored of 0 calls; L-insight-OE020: 0 scored of 0 calls; C-insight-OE020: 0 scored of 0 calls; L-outsight-OE055: 0 scored of 0 calls; C-outsight-OE055: 0 scored of 0 calls; L-outsight-OE028: 0 scored of 0 calls; C-outsight-OE028: 0 scored of 0 calls; L-outsight-OE062: 0 scored of 0 calls; C-outsight-OE062: 0 scored of 0 calls. Not imputed.
