Budget episodes: 171. RoPE setting recorded in run.json: {'native': 171}.

| dimension | budget eps | eps with any retrieval | median chunks fired / total | median length s, budget | median length s, stock | budget eps <= 4.5 s |
|---|---:|---:|---:|---:|---:|---:|
| gc | 48 | 43 | 4 / 8 | 8.3 | 8.8 | 5 |
| if | 50 | 34 | 1 / 5 | 4.8 | 4.8 | 16 |
| insight | 30 | 30 | 56 / 60 | 59.8 | 59.8 | 0 |
| outsight | 43 | 29 | 2 / 6 | 5.8 | 5.8 | 14 |

Paired cases (both arms scored): 170 of 171. Unscored in either arm: [('insight', 'OE013')].

| slice | pairs | stock mean | budget mean | paired delta [95% CI] | W/T/L |
|---|---:|---:|---:|---|---|
| all | 170 | 1.88 | 1.83 | -0.05 [-0.25, +0.14] | 41/86/43 |
| retrieval fired (>= 1 chunk) | 135 | 1.92 | 1.77 | -0.15 [-0.38, +0.08] | 35/59/41 |
| retrieval never fired | 35 | 1.75 | 2.05 | +0.30 [+0.02, +0.65] | 6/27/2 |
| fired in >= 5 chunks | 68 | 1.76 | 1.82 | +0.05 [-0.30, +0.40] | 26/24/18 |
| gc: all | 48 | 2.41 | 2.37 | -0.04 [-0.46, +0.39] | 16/17/15 |
| gc: fired | 43 | 2.32 | 2.27 | -0.05 [-0.52, +0.43] | 15/13/15 |
| gc: not fired | 5 | 3.20 | 3.24 | +0.04 [+0.00, +0.12] | 1/4/0 |
| if: all | 50 | 1.84 | 1.76 | -0.08 [-0.43, +0.25] | 10/27/13 |
| if: fired | 34 | 1.86 | 1.55 | -0.31 [-0.74, +0.10] | 6/17/11 |
| if: not fired | 16 | 1.80 | 2.19 | +0.39 [-0.05, +0.92] | 4/10/2 |
| insight: all | 29 | 1.68 | 1.74 | +0.06 [-0.50, +0.58] | 13/8/8 |
| insight: fired | 29 | 1.68 | 1.74 | +0.06 [-0.49, +0.58] | 13/8/8 |
| insight: not fired | 0 | n/a | n/a | n/a | n/a |
| outsight: all | 43 | 1.47 | 1.36 | -0.11 [-0.38, +0.18] | 2/34/7 |
| outsight: fired | 29 | 1.61 | 1.31 | -0.30 [-0.60, -0.04] | 1/21/7 |
| outsight: not fired | 14 | 1.18 | 1.46 | +0.29 [+0.00, +0.86] | 1/13/0 |
| earlier powered cohort (14 cases), in this run | 14 | 1.59 | 1.62 | +0.04 [-0.52, +0.73] | 4/6/4 |
| powered cohort: GC | 7 | 1.60 | 2.14 | +0.54 [-0.29, +1.74] | 4/2/1 |
| powered cohort: out-of-sight | 7 | 1.57 | 1.11 | -0.46 [-1.04, -0.04] | 0/4/3 |
| powered cohort: 7 long-plan cases | 7 | 1.57 | 1.37 | -0.20 [-0.57, +0.17] | 1/4/2 |
| powered cohort: 7 original cases | 7 | 1.60 | 1.88 | +0.28 [-0.81, +1.62] | 3/2/2 |

| case | stock | budget | chunks fired / total | budget s | stock s | stock gate | budget gate |
|---|---:|---:|---:|---:|---:|---|---|
| GC005 | 1.0 | 1.0 | 15/19 | 18.8 | 16.8 | fail | fail |
| GC015 | 3.0 | 2.0 | 8/12 | 11.8 | 13.8 | pass | partial |
| GC017 | 2.0 | 2.2 | 5/9 | 8.8 | 8.8 | pass | partial |
| GC019 | 1.8 | 2.4 | 6/10 | 9.8 | 10.8 | pass | pass |
| GC024 | 1.0 | 1.2 | 4/8 | 7.8 | 6.8 | pass | pass |
| GC025 | 1.2 | 1.2 | 7/11 | 10.8 | 10.8 | pass | pass |
| GC027 | 1.2 | 5.0 | 11/15 | 14.8 | 8.8 | pass | pass |
| OE026 | 2.0 | 1.75 | 1/5 | 4.8 | 4.8 | pass | pass |
| OE029 | 1.0 | 1.0 | 1/5 | 4.8 | 4.8 | pass | pass |
| OE044 | 2.0 | 1.0 | 9/13 | 12.8 | 12.8 | partial | pass |
| OE055 | 1.0 | 1.0 | 6/10 | 9.8 | 10.8 | fail | pass |
| OE056 | 1.0 | 1.0 | 6/10 | 9.8 | 9.8 | fail | pass |
| OE057 | 1.0 | 1.0 | 0/4 | 3.8 | 3.8 | fail | fail |
| OE069 | 3.0 | 1.0 | 4/8 | 7.8 | 6.8 | pass | partial |

Gate verdicts, paired cases: stock {'partial': 25, 'pass': 81, 'fail': 32, 'not_applicable': 32}; budget {'fail': 27, 'pass': 84, 'partial': 27, 'not_applicable': 32}.
Score exactly 1.0: stock 86, budget 89 of 170.
Mean episode length: stock 16.2 s, budget 16.2 s; pairs with identical length: 100 of 170.

Identical-pixel pairs (every chunk's pixel hash equal in both arms): 28. Pairs scored differently by the judge: 5; mean absolute gap 0.35, largest 4.00; mean signed budget-minus-stock +0.35.

| identical-pixel pair scored differently | stock | budget |
|---|---:|---:|
| gc-GC033-seed42 | 4.2 | 4.4 |
| if-IF021-seed42 | 1.0 | 3.0 |
| if-IF043-seed42 | 2.0 | 5.0 |
| if-IF050-seed42 | 2.33 | 3.0 |
| outsight-OE046-seed42 | 1.0 | 5.0 |
