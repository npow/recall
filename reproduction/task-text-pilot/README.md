# Task text conditioning: development pilot

Supplying the task objective alongside the image caption improved the four-case IF development mean from 1.00 to 1.715. This is a small development result, not evidence of beating the published overall score.

| Case | Caption | Objective | Caption + objective |
|---|---:|---:|---:|
| IF001 | 1.00 | 1.00 | 1.00 |
| IF025 | 1.00 | 2.00 | 2.86 |
| IF021 | 1.00 | 1.00 | 2.00 |
| IF042 | 1.00 | 1.00 | 1.00 |
| Mean | 1.00 | 1.25 | 1.715 |

All twelve declared episodes completed with adaptive player decisions, seed 42, a 40-chunk budget, the native 18/6 cache, H200/Torch 2.6 and unchanged streamed Haiku player requests. The four cases came from the existing declared cohort, rather than selection by this experiment's outcomes. No weights were trained. Runs naturally ended after 3–5 chunks; this is not a long-horizon memory test.

The released Gemini verifier ran through Vertex using `gemini-3.1-pro-preview`. Served model versions are retained where returned. The alias and our player/adapter differences prevent claiming an exact hosted-leaderboard reproduction.

## Where it breaks

IF001 with objective text makes the woman walk, but she passes through the bus-stop glass, bench and hedge. The interaction-quality cap keeps its score at 1. IF021 with combined text moves the waiter but loses the tray and drinks, capping quality at 2. IF042 retains incorrect jumping/floating rather than grounded walking. Better task conditioning does not fix physical dynamics.

Inspect [the IF001 comparison](IF001-contact.jpg), individual `rollout.mp4` files, their `gemini-score.json` judgments, and [summary.json](summary.json). Raw job outcomes, task/image hashes, generation settings and failure records are retained.

## Confirmation

[The frozen confirmation plan](../task-text-confirmation/plan.json) compares caption versus combined text on all four development cases with seed 43 and four additional cases selected by a fixed hash before judging. Both partitions must be reported separately. The candidate has not passed that confirmation yet.

Regenerate scores with `.venv-lingbot/bin/python scripts/evaluate_task_text_pilot.py`. Existing judgments are reused only when the video hash matches. Generation failures remain in the declared cohort; evaluator failures prevent a complete mean.
