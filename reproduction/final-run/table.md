# final1: stock vs static text director (frozen config: reproduction/final-run-config.json)

Overall = equal mean of the four dimension means; 95% CI by bootstrap over cases within each dimension.
Local checkpoint, Lambda H100, Haiku player, Gemini `gemini-3.1-pro-preview`; see the config for deviations.

## primary single call

| set | arm | GC | IF | insight | OE | overall [95% CI] | n (GC/IF/insight/OE) |
|---|---|---:|---:|---:|---:|---|---|
| all_171 | stock | 2.36 | 1.62 | 1.81 | 1.44 | 1.81 [1.64, 1.99] | 48/50/30/43 |
| all_171 | director | 2.36 | 1.69 | 1.79 | 1.45 | 1.82 [1.66, 2.01] | 48/50/30/43 |
| all_171 | director - stock | +0.00 | +0.07 | -0.01 | +0.01 | +0.02 [-0.15, +0.19] | paired |
| held_out_151 | stock | 2.33 | 1.58 | 1.68 | 1.45 | 1.76 [1.58, 1.94] | 45/45/24/37 |
| held_out_151 | director | 2.33 | 1.71 | 1.73 | 1.52 | 1.82 [1.65, 2.01] | 45/45/24/37 |
| held_out_151 | director - stock | +0.00 | +0.13 | +0.04 | +0.08 | +0.06 [-0.13, +0.26] | paired |
| all_without_GC013 | stock | 2.37 | 1.62 | 1.81 | 1.44 | 1.81 [1.64, 2.00] | 47/50/30/43 |
| all_without_GC013 | director | 2.37 | 1.69 | 1.79 | 1.45 | 1.83 [1.66, 2.00] | 47/50/30/43 |
| all_without_GC013 | director - stock | +0.00 | +0.07 | -0.01 | +0.01 | +0.02 [-0.15, +0.19] | paired |
| held_out_without_GC013 | stock | 2.34 | 1.58 | 1.68 | 1.45 | 1.76 [1.58, 1.95] | 44/45/24/37 |
| held_out_without_GC013 | director | 2.34 | 1.71 | 1.73 | 1.52 | 1.82 [1.65, 2.01] | 44/45/24/37 |
| held_out_without_GC013 | director - stock | +0.00 | +0.13 | +0.04 | +0.08 | +0.06 [-0.13, +0.25] | paired |

## secondary mean of 3

| set | arm | GC | IF | insight | OE | overall [95% CI] | n (GC/IF/insight/OE) |
|---|---|---:|---:|---:|---:|---|---|
| all_171 | stock | incomplete | | | | | |
| all_171 | director | incomplete | | | | | |
| held_out_151 | stock | incomplete | | | | | |
| held_out_151 | director | incomplete | | | | | |
| all_without_GC013 | stock | incomplete | | | | | |
| all_without_GC013 | director | incomplete | | | | | |
| held_out_without_GC013 | stock | incomplete | | | | | |
| held_out_without_GC013 | director | incomplete | | | | | |

Published references: LingBot-World2 hosted 1.82 (GC 2.04, IF 2.13, insight 1.95, OE 1.16); Genie 3 2.12.
