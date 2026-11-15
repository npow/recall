# Judge-repeat sensitivity analysis (Gemini, 3 calls per episode)

Sensitivity analysis only. The frozen single-call results in `reproduction/full-run/` stay the primary numbers.
Out-of-sight and GC episodes of the full run (both arms) that scored exactly 1.0 or had a gate verdict of fail / partial were re-judged with 2 extra independent Gemini calls (same released prompts, rubric, gate, sampling and model; separate caches). **Selection is one-sided**: floor episodes can only go up, and gate-affected ones were chosen because their first call was low, so the repeated means are upper-biased (regression to the mean). This measures whether the judge is *consistent*, not whether it is *right*.

Episodes with both repeats: 118.

| arm | dimension | n | single-call mean | mean of 3 | mean per-episode SD | majority-gate flips | lower | higher |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stock | GC | 22 | 1.69 | 1.65 | 0.44 | 2 | 4 | 11 |
| stock | out-of-sight | 35 | 1.14 | 1.16 | 0.16 | 3 | 2 | 4 |
| budget | GC | 22 | 1.64 | 1.72 | 0.47 | 1 | 5 | 8 |
| budget | out-of-sight | 39 | 1.22 | 1.24 | 0.20 | 6 | 2 | 5 |

Out-of-sight floor: 60 of 69 episodes that scored exactly 1.0 score 1.0 on all three calls.

Effect on the full-run table when the repeated episodes use their mean of 3 (others keep their single call):

| | GC | IF | insight | out-of-sight | overall |
|---|---:|---:|---:|---:|---:|
| stock, single call (primary) | 2.41 | 1.84 | 1.68 | 1.47 | 1.85 |
| budget, single call (primary) | 2.37 | 1.76 | 1.72 | 1.36 | 1.80 |
| stock, with repeats | 2.39 | 1.84 | 1.68 | 1.49 | 1.85 |
| budget, with repeats | 2.41 | 1.76 | 1.72 | 1.38 | 1.82 |

Paired budget - stock (overall = equal mean of the four dimension deltas, bootstrap over cases):

* single call: -0.04 [-0.26, +0.16]; GC -0.04, IF -0.08, insight +0.06, out-of-sight -0.11
* with repeats: -0.03 [-0.22, +0.17]; GC +0.02, IF -0.08, insight +0.06, out-of-sight -0.11

Per-episode values: [per-episode.csv](per-episode.csv). Machine-readable summary: [summary.json](summary.json).
