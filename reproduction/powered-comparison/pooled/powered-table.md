| Arm | n | Gemini mean [95% CI] | GC | OE | gate non-fail | Opus mean |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| stock | 28 | 1.81 [1.35, 2.35] | 2.46 | 1.16 | 21/28 | 2.17 |
| budget | 28 | 2.13 [1.66, 2.67] | 2.81 | 1.45 | 25/28 | 2.33 |
| cache46 | 28 | 2.26 [1.76, 2.83] | 2.53 | 2.00 | 21/28 | 2.28 |

| Paired vs stock | slice | pairs | mean delta [95% CI] | W/T/L |
| --- | --- | ---: | --- | --- |
| budget | gemini_all | 28 | 0.32 [-0.08, 0.75] | 9/14/5 |
| budget | gemini_gc | 14 | 0.36 [-0.39, 1.13] | 6/5/3 |
| budget | gemini_oe | 14 | 0.29 [-0.03, 0.71] | 3/9/2 |
| budget | opus_all | 26 | 0.09 [-0.05, 0.23] | 13/8/5 |
| cache46 | gemini_all | 28 | 0.46 [-0.22, 1.15] | 12/10/6 |
| cache46 | gemini_gc | 14 | 0.07 [-1.07, 1.21] | 7/2/5 |
| cache46 | gemini_oe | 14 | 0.84 [0.20, 1.62] | 5/8/1 |
| cache46 | opus_all | 26 | 0.05 [-0.08, 0.18] | 8/11/7 |
