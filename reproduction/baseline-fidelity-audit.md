# Baseline-fidelity audit (independent, read-only)

Run on 2026-11-15 with a 44-agent workflow: five independent lenses (protocol parity, judge input fidelity, player parity, world-model interface, statistics), each claimed deviation then checked by a separate skeptic that tried to refute it. No GPU was used. This is the synthesizer's verdict, unedited except for this header; it reports our stock (local checkpoint, Gemini judge) arm against the paper's hosted-service LingBot-World2 numbers. Numbers were computed on the full-run results available when the audit ran (scoring was still finishing), so final figures live in LEADERBOARD.md.

**Baseline-fidelity verdict for the LingBot-World2 stock arm (local checkpoint vs. hosted paper numbers)**

This is a synthesis of the verified findings only. I ran no new checks.

**Bottom line:** The pipeline is faithful on the judge input, protocol, player and insight handling. No bug is found that would explain a large gap. The stock arm is statistically compatible with the paper on GC, insight and out-of-sight, but the n≈30–48 confidence intervals (±0.4–0.45) are too wide to call it a tight reproduction. IF is the only dimension where we are plausibly below the paper. The one finding that could matter is the judge's false "static" verdicts.

**(1) Per-dimension verdict** (ours = Gemini 3.1 Pro with the released rubric, gate and aggregation)

| Dimension | Ours | Paper | Verdict |
|---|---|---|---|
| GC | 2.41 (CI 2.02–2.83, n=48); 2.37 without GC013 | 2.04 | Consistent, with ours at the high edge (z≈+1.8). We are not below the paper. Not a clean match either: ours is higher, and the player/interface differences likely inflate it. |
| IF | 1.77 final (about 2.0 in earlier runs) | 2.13 | Not reproduced, or cannot tell. It is the only dimension where we are lower. The CI for this dimension was not reported to me. Known causes: the gate (some of the verdict failures are false-static), U/J keys dropped, and 12 IF plans needing interact() not attempted. One finding says unattempted interact() plans depress IF by roughly 0.05–0.07. The IF deficit is therefore not fully explained. |
| Insight | 1.68 caption-only (CI 1.30–2.14, n=29) | 1.95 | Consistent, with a caveat. This is a different prompt from the released adapters, which send "Task: wait and observe". The 'both' result (about 1.2) is the released-style prompt, and it is below the paper. The paper-faithful prompt scores worse, so we cannot claim insight reproduction under the released prompt. Our 1.68 relies on a deviation we chose after a 6-case dev test. |
| Out-of-sight | 1.47 (CI 1.19–1.80, n=43) | 1.16 | Consistent, with ours higher. 33/43 are at exactly 1.0, so the benchmark is floor-dominated. Only 10/43 (23%) are instruction-gate failures. The other 23 are capped by the rubric rule for static/no meaningful evolution. The 26% "gate failure" figure in the brief is really 23%, and it does not explain the score. |
| Overall | Not computed | 1.82 | Cannot tell. Per-dimension noise of ±0.4 puts any overall comparison within the error bars. |

**What does look wrong:** the judge calls OE056, OE043 and OE041 "completely static" at confidence 5, while the sampled grids visibly show the commanded motion. This was verified by inspecting the sheets. The judge input itself is clean: sampling, encoding, frame counts, hashes and aspect ratio all match the released code. This points at the judge's reading of slow pans, not at our pipeline.

Open items:
- It is unproven whether the paper's hosted-service videos have the same issue.
- No Opus scores exist for the full run, so we have no second judge.
- Our rotation and speed constants (60°/s, move_speed 1.0) are reconstructed, not verified against the hosted service.

**(2) Deviations ranked by likely score impact**

| # | Deviation | Direction | Cheapest check or fix | GPU cost |
|---|---|---|---|---|
| 1 | Judge false-"static" verdicts on weak-signal pans (OE041/043/056/060) | Lowers OE and IF | Re-judge a sample of the 23 floor episodes with 3 samples per gate. Optionally add an Opus second opinion on the same videos. Report single-call and majority-vote scores side by side. | None (Gemini calls only) |
| 2 | Insight prompt: caption-only vs released "Task: wait and observe" | Raises insight; 'both' scores about 1.2 vs caption-only 1.68 | Report both variants. A re-roll of 'both' is already done (about 1.2). | None |
| 3 | U/J (pitch) keys dropped by the plan parser (GC008, GC016, OE028/030/032/036/047) | Lowers (OE032 has a gate failure with no tilt) | Add `'u':'UP','j':'DOWN'` to PLAN_ACTIONS at reference_player.py:41/139. Rerun only those 7 tasks. | About 7 episodes, small |
| 4 | interact() steps for 12 IF plans not attempted | Lowers IF slightly | Cannot cheaply fix without interaction keys. Report as a known gap. | N/A |
| 5 | Plan-strict player: END challenge, plan-length rollouts (median 1.06× plan) | Unclear | Report as is. The ablation would need new rollouts. | Moderate (rollouts) |
| 6 | Dataset inconsistency OE059 (prompt says "left", plan starts D) | Lowers 1 task | Exclude it in a sensitivity analysis, or report as is. | None |
| 7 | GC013 invalid image/caption pair | Raises GC by about 0.04–0.06 | Report with and without it (2.41 / 2.37). | None |
| 8 | Hosted vs local model: 480x832 vs 720p, KV-cache scheduling, no director | Unknown | Cannot be fixed. State it as an uncontrolled difference. | N/A |

**Recommended order:** run item 1 first because it is free and addresses the biggest worry. Item 3 is cheap but affects only 7 tasks, so it will not move the means much. Items 2, 6 and 7 are free reporting changes.

**(3) What to say about baseline fidelity in the submission**

- Do not claim "we reproduced the published numbers." Say: "We ran the open local checkpoint under the released player, rubric, gate and aggregation. Scores per dimension are GC 2.41, IF 1.77, insight 1.68 (1.2 with the released prompt), out-of-sight 1.47. Bootstrap CIs of about ±0.4 overlap the paper's hosted-service values on GC, insight and out-of-sight. IF is lower. These are statistically compatible with, but not an exact reproduction of, the published values."
- State the deviations explicitly: local checkpoint rather than hosted service, a Haiku 4.5 player, the caption-only insight prompt, the U/J parsing bug (fixed or not), the lack of interact() support, and the reconstructed motion constants.
- State that 77% of out-of-sight episodes sit at the 1.0 floor for every system. Differences smaller than about 0.4 per dimension are not detectable.
- Report that the judge shows false "static" verdicts on slow pans. Give single-call scores, and add voted scores if item 1 is run.
- Do not claim that our higher GC or out-of-sight numbers indicate a better model.

Key file paths:
- /home/npow/code/iteratehack/.claude/worktrees/agent-a20dd1a4d88b1a928/worldsync/reference_player.py
- /home/npow/code/iteratehack/.claude/worktrees/agent-a20dd1a4d88b1a928/reproduction/full-run-config.json
- /home/npow/code/iteratehack/.claude/worktrees/agent-a20dd1a4d88b1a928/outputs/adaptive/full1
