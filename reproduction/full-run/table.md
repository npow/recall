| | GC | IF | insight | out-of-sight | overall (equal mean of four) |
|---|---:|---:|---:|---:|---:|
| paper, LingBot-World2 (hosted) | 2.04 | 2.13 | 1.95 | 1.16 | 1.82 |
| paper, Genie 3 (closed) | | | | | 2.12 |
| ours stock, all 171 cases (scored 48/50/29/43) | 2.41 | 1.84 | 1.68 | 1.47 | 1.85 [1.67, 2.04] |
| ours budget, all 171 cases (scored 48/50/30/43) | 2.37 | 1.76 | 1.72 | 1.36 | 1.80 [1.63, 1.98] |
| ours stock, without GC013 (scored 47/50/29/43) | 2.37 | 1.84 | 1.68 | 1.47 | 1.84 [1.67, 2.03] |
| ours budget, without GC013 (scored 47/50/30/43) | 2.31 | 1.76 | 1.72 | 1.36 | 1.79 [1.62, 1.97] |
| ours stock, insight = 24 held-out cases (6 dev cases dropped) (scored 48/50/23/43) | 2.41 | 1.84 | 1.82 | 1.47 | 1.89 [1.70, 2.09] |
| ours budget, insight = 24 held-out cases (6 dev cases dropped) (scored 48/50/24/43) | 2.37 | 1.76 | 1.63 | 1.36 | 1.78 [1.62, 1.96] |

Paired budget - stock (cases scored in both arms; overall = equal mean of the four dimension deltas):

* all 171: overall -0.04 [-0.26, 0.16]; GC -0.04 (n=48), IF -0.08 (n=50), insight +0.06 (n=29), out-of-sight -0.11 (n=43)
* without GC013: overall -0.05 [-0.26, 0.15]; GC -0.06 (n=47), IF -0.08 (n=50), insight +0.06 (n=29), out-of-sight -0.11 (n=43)
* insight 24 held-out: overall -0.10 [-0.32, 0.12]; GC -0.04 (n=48), IF -0.08 (n=50), insight -0.16 (n=23), out-of-sight -0.11 (n=43)

Gemini gate outcomes and the score-1.0 floor, per arm and dimension:

| arm/dimension | scored | gate fail | partial | pass | score exactly 1.0 | revisit required | revisit observed |
|---|---:|---:|---:|---:|---:|---:|---:|
| stock/gc | 48 | 4 | 13 | 31 | 13 | 38 | 29 |
| stock/if | 50 | 18 | 6 | 23 | 24 | 0 | 0 |
| stock/insight | 29 | 0 | 0 | 0 | 16 | 0 | 0 |
| stock/outsight | 43 | 10 | 6 | 27 | 33 | 39 | 28 |
| budget/gc | 48 | 5 | 11 | 32 | 13 | 38 | 31 |
| budget/if | 50 | 16 | 8 | 23 | 28 | 0 | 0 |
| budget/insight | 30 | 0 | 0 | 0 | 13 | 0 | 0 |
| budget/outsight | 43 | 6 | 8 | 29 | 36 | 39 | 32 |

Kept separately as a 'both'-prompt sensitivity datum (not in any table above): budget-insight-OE025-seed42

Generation failures after retries: {'stock': 0, 'budget': 0}. Episodes generated but not scored (judge errors): {'stock': 1, 'budget': 0}.
Expected episodes per arm and dimension: {"gc": 48, "if": 50, "insight": 30, "outsight": 43}.

A claim of beating 2.12 requires the lower CI bound of the overall above 2.12 (reproduction/pass-criterion.md).
