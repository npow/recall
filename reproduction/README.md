# Local baseline before hillclimbing

**Current target: LingBot-World2 14B causal-fast. Baseline not measured yet.**

The published **1.82** belongs to the hosted LingBot service. It is a reference,
not an expected score for the public local checkpoint or our Claude evaluator.
We will establish our own reproducible stock baseline before changing inference
or memory. The [state file](state.json) records the closed gate; matching the
hosted number is not required to verify the local baseline.

## 1. Freeze and measure the stock model

Pin checkpoint, source, input dataset, resolution, sampler, cache, anchors, seed,
player, judge, scoring policy and action timing. Use the official unquantized 14B
causal-fast release. Preserve videos, native controls, player decisions, actual
call counts, terminations and errors for every case. Score all 171 cases and
report coverage per dimension; missing generations or judge errors cannot
silently become scores or disappear from the denominator.

Claude with the released PlayWorld rubric is the authorized local evaluator.
Numerical comparisons with the published Gemini table remain qualified. No
Gemini calls are made. See [setup and deviations](lingbot.md),
[asset inventory](assets.json), and [live progress](lingbot-progress.json).

## 2. Verify the persistent control adapter

Match native inputs and native generation before interpreting quality changes.
Verify pose continuity, fixed movement and turn rates, and neutral camera inputs
during WAIT. Check realized forward/backward motion, turns and revisits in the
generated observations. Intended poses alone do not prove successful navigation.
Separate integration errors, model control failures, identity/geometry failures,
and evolution freezes or resets. Do not label every low score as forgetting.

Native/adapter pixel parity, source camera-tensor parity, and native VAE streaming
parity are integration checks. They do not establish the hosted score. Complete
scored episodes and realized trajectory evidence are also required.

## 3. Tune existing inference knobs

Only after the local stock baseline and controls are verified, compare cache
length, retained initial anchors, and player decision intervals. Select using
complete scored development episodes; report player-call and inference costs.
Evaluate the chosen settings on held-out episodes under the frozen evaluator.
Keep the baseline results and all failed trials.

## 4. Retrieve history only for demonstrated forgetting

If controls work and revisited regions lose identity or geometry, test historical
view/feature retrieval within the v2 cache. The earlier training-free study used
LingBot-World Fast, not this v2 checkpoint, so transfer is a hypothesis to test.

## 5. Fine-tune only for a repeatable failure

If valid control still produces process freezes or resets, do not assume longer
cache fixes it. Gather separate training examples and fine-tune an adapter only
following diagnosis. Keep improvements that survive held-out, player-driven
episodes. No training from scratch, image editing, reward-model work or RL is
part of the current implementation plan.

The [paper/release audit](../docs/paper-protocol-audit.md),
[Matrix integration](matrix-adapter.md), and [earlier ABot work](../docs/exploratory-setup.md)
remain historical evidence. Neither the Matrix smoke nor the exploratory ABot
scores count as the LingBot local baseline.
