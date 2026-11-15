| Arm | n | Gemini mean [95% CI] | GC | OE | gate non-fail | Opus mean |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| stock | 14 | 1.89 [1.29, 2.68] | 2.57 | 1.21 | 10/14 | 2.25 |
| budget | 14 | 2.29 [1.60, 3.07] | 2.83 | 1.75 | 13/14 | 2.27 |
| cache46 | 14 | 1.94 [1.37, 2.67] | 2.31 | 1.57 | 10/14 | 2.29 |

| Paired vs stock | slice | pairs | mean delta [95% CI] | W/T/L |
| --- | --- | ---: | --- | --- |
| budget | gemini_all | 14 | 0.40 [-0.03, 0.88] | 4/7/3 |
| budget | gemini_gc | 7 | 0.26 [-0.17, 0.94] | 2/3/2 |
| budget | gemini_oe | 7 | 0.54 [-0.07, 1.14] | 2/4/1 |
| budget | opus_all | 14 | 0.02 [-0.21, 0.25] | 6/3/5 |
| cache46 | gemini_all | 14 | 0.05 [-0.79, 0.81] | 5/5/4 |
| cache46 | gemini_gc | 7 | -0.26 [-1.77, 1.20] | 3/1/3 |
| cache46 | gemini_oe | 7 | 0.36 [-0.07, 0.96] | 2/4/1 |
| cache46 | opus_all | 14 | 0.04 [-0.15, 0.24] | 4/7/3 |
