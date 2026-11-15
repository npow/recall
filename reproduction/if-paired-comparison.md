# Paired IF comparison: canonical controller vs baseline player

**43 of 49 supported IF tasks complete** (both arms, 3 Gemini calls each). Seed 42; each pair on one GPU.

## Result

| | Baseline | Canonical controller | Δ |
|---|---:|---:|---:|
| IF mean | 1.73 | 1.74 | 0.01 |
| Gate pass (calls) | 49 | 55 | |
| Gate fail (calls) | 54 | 40 | |

Paired mean difference 0.01, task-bootstrap 95% interval [-0.30, 0.32]; 18 better / 10 equal / 15 worse; one-sided sign test p = 0.364.

## Per task

| Task | Baseline calls | Ours calls | Δ | Baseline gates | Ours gates |
|---|---|---|---:|---|---|
| IF012 | 1.0, 1.0, 1.0 | 1.0, 1.8, 1.0 | +0.27 | fail, fail, fail | fail, pass, fail |
| IF006 | 2.0, 2.0, 2.0 | 2.4, 2.0, 1.7 | +0.03 | partial, partial, partial | pass, pass, pass |
| IF022 | 1.0, 1.0, 1.0 | 3.0, 2.0, 2.0 | +1.33 | fail, fail, fail | partial, partial, partial |
| IF007 | 1.0, 1.0, 1.0 | 1.0, 1.0, 1.0 | +0.00 | fail, fail, fail | fail, fail, fail |
| IF043 | 1.0, 1.0, 1.0 | 1.0, 1.0, 1.0 | +0.00 | pass, fail, partial | pass, pass, partial |
| IF011 | 1.0, 1.0, 1.0 | 1.0, 1.0, 1.0 | +0.00 | fail, fail, fail | fail, fail, fail |
| IF020 | 1.0, 1.0, 1.6 | 2.0, 2.0, 2.0 | +0.80 | fail, fail, partial | partial, partial, partial |
| IF013 | 1.0, 1.0, 1.0 | 1.0, 1.0, 1.0 | +0.00 | fail, fail, fail | fail, fail, fail |
| IF024 | 2.7, 2.0, 1.6 | 1.0, 1.0, 1.0 | -1.09 | pass, pass, pass | fail, fail, fail |
| IF027 | 2.0, 2.0, 2.0 | 1.0, 1.0, 1.0 | -1.00 | pass, pass, pass | pass, partial, partial |
| IF014 | 3.0, 2.0, 1.0 | 1.0, 1.0, 1.0 | -1.00 | pass, pass, pass | fail, fail, fail |
| IF048 | 1.0, 1.0, 1.0 | 1.0, 1.0, 1.0 | +0.00 | fail, fail, fail | fail, fail, fail |
| IF038 | 2.0, 3.0, 1.0 | 3.0, 2.0, 1.0 | +0.00 | partial, partial, partial | partial, partial, fail |
| IF049 | 2.0, 3.9, 3.0 | 1.0, 1.0, 1.0 | -1.95 | partial, pass, pass | fail, fail, fail |
| IF042 | 4.0, 5.0, 1.0 | 3.0, 3.0, 3.0 | -0.33 | pass, pass, pass | pass, pass, pass |
| IF028 | 2.0, 1.0, 1.0 | 1.0, 1.0, 1.0 | -0.33 | partial, fail, fail | fail, fail, fail |
| IF026 | 2.0, 2.8, 2.5 | 1.8, 2.7, 3.0 | +0.06 | partial, partial, partial | pass, pass, pass |
| IF030 | 4.0, 2.5, 4.0 | 1.0, 1.0, 1.0 | -2.50 | pass, pass, pass | partial, pass, pass |
| IF015 | 4.0, 3.0, 4.7 | 3.0, 3.0, 2.7 | -1.00 | pass, pass, pass | pass, pass, pass |
| IF023 | 1.0, 1.0, 1.0 | 1.0, 2.0, 2.0 | +0.67 | fail, fail, fail | partial, partial, partial |
| IF032 | 1.0, 1.0, 1.0 | 1.8, 1.8, 2.0 | +0.87 | fail, fail, fail | pass, pass, pass |
| IF051 | 3.0, 2.0, 2.0 | 1.0, 1.0, 2.4 | -0.86 | pass, pass, pass | pass, pass, pass |
| IF050 | 3.0, 2.7, 3.0 | 2.7, 2.2, 2.8 | -0.33 | pass, pass, pass | pass, pass, pass |
| IF029 | 2.0, 1.0, 1.0 | 3.0, 1.0, 2.7 | +0.89 | pass, pass, pass | partial, fail, pass |
| IF019 | 3.0, 2.0, 3.0 | 5.0, 2.0, 2.0 | +0.33 | not_applicable, not_applicable, not_applicable | not_applicable, not_applicable, not_applicable |
| IF031 | 1.0, 1.0, 1.0 | 1.0, 1.0, 1.0 | +0.00 | fail, fail, fail | fail, fail, fail |
| IF016 | 1.0, 1.0, 1.0 | 1.8, 1.0, 2.0 | +0.60 | pass, pass, pass | pass, partial, pass |
| IF036 | 4.0, 4.0, 3.0 | 1.8, 2.0, 1.0 | -2.06 | pass, pass, pass | pass, pass, fail |
| IF035 | 3.0, 4.0, 3.0 | 3.0, 2.0, 2.3 | -0.90 | pass, pass, pass | pass, pass, pass |
| IF025 | 2.0, 1.0, 1.7 | 2.0, 1.0, 1.0 | -0.22 | pass, pass, pass | pass, pass, pass |
| IF021 | 1.0, 3.0, 2.0 | 1.0, 1.0, 1.3 | -0.89 | fail, pass, pass | fail, pass, pass |
| IF053 | 1.0, 1.0, 1.0 | 1.0, 1.0, 1.0 | +0.00 | fail, fail, fail | pass, pass, pass |
| IF017 | 1.0, 1.0, 1.0 | 1.0, 1.0, 1.0 | +0.00 | fail, fail, fail | fail, fail, fail |
| IF002 | 1.0, 1.0, 1.0 | 2.0, 2.0, 3.0 | +1.33 | fail, fail, fail | partial, partial, partial |
| IF009 | 1.0, 1.0, 1.0 | 3.0, 2.0, 2.0 | +1.33 | fail, fail, fail | pass, partial, partial |
| IF037 | 1.0, 1.0, 2.0 | 1.7, 2.9, 2.4 | +0.99 | partial, partial, partial | partial, pass, pass |
| IF010 | 2.0, 1.8, 2.0 | 1.0, 1.0, 2.0 | -0.61 | pass, pass, partial | partial, pass, pass |
| IF045 | 1.0, 1.0, 1.0 | 1.0, 1.0, 1.0 | +0.00 | fail, fail, fail | fail, fail, fail |
| IF052 | 1.0, 1.0, 1.0 | 1.5, 1.5, 1.0 | +0.33 | fail, fail, fail | pass, pass, pass |
| IF003 | 1.0, 2.0, 1.0 | 5.0, 5.0, 5.0 | +3.67 | not_applicable, not_applicable, not_applicable | not_applicable, not_applicable, not_applicable |
| IF034 | 1.0, 1.0, 1.6 | 1.4, 1.4, 3.0 | +0.73 | pass, pass, pass | pass, pass, pass |
| IF047 | 1.0, 1.0, 1.0 | 2.0, 1.0, 2.0 | +0.67 | fail, fail, fail | partial, fail, pass |
| IF041 | 1.8, 2.0, 2.0 | 3.0, 2.7, 1.8 | +0.55 | partial, partial, partial | partial, partial, partial |

Tasks where every corrected call beats every baseline call (demo candidates): IF003 (1.33→5.00), IF022 (1.00→2.33), IF002 (1.00→2.33), IF009 (1.00→2.33), IF032 (1.00→1.87), IF020 (1.20→2.00).

## Setup and caveats

- Baseline = frozen full-run configuration (`5b8a5b8`, `--text-prompt both`, 40 chunks, Haiku 4.5 player).
- Canonical controller = `lingbot_released_baseline --generate --validate-controller` on main, 60 chunks, hold-service token cap 400.
- Compares two complete configurations (episode length, text prompt and controller all differ), not an isolated ablation.
- Judge: `gemini-3.1-pro-preview` on Vertex with the released rubric; the paper's exact model id is unavailable.
- GPUs: Lambda H100 SXM and the local RTX PRO 6000; both arms of a task always ran on the same GPU.
