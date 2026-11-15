# Adaptive four-view pilot

Both frozen stock seeds are complete and scored before launching memory. Each
produced 12.8125seconds, with an automatic neutral bootstrap followed by twelve
RIGHT commands at30degrees/second, then player END. Commanded yaw is360degrees;
it is not a measurement of the generated camera trajectory.

| Seed | Stock Claude rubric | Visual trajectory gate | Revisit observed |
|---|---:|---|---|
|42|3.4|partial|no|
|43|3.8|partial|no|

The mean3.6 is a local rubric score, not a leaderboard result. Both episodes
fail the requested visual return despite the player's confident completion
claim. The endpoint observation shown to the player agrees with the assembled
video endpoint (mean absolute pixel differences~2.7/255 from separate JPEG/video
compression), so this does not appear to be a stale observation integration bug.
See the completion audit, contact sheet and complete videos.

The corrected generation's local Modal entrypoint exited143 while remote tasks
continued. Both committed run artifacts were recovered from the same live app;
rollout, controller and player source hashes match the frozen source. Neither
remote generation was restarted. The earlier SDK-temperature failures are also
retained in failed-initial/.

Memory candidates use exactly the same pinned generation source660fa4e, inputs,
controller, seeds, H200 hardware and Claude models. The selected method remains
four-nearest early-view memory with the original18-frame attention budget.
All memory candidates are now complete and scored:

| Seed | Stock rubric | Four-view rubric | Memory gate | Memory revisit |
|---|---:|---:|---|---|
|42|3.4|3.8|partial|no|
|43|3.8|4.0|partial|no|

The mean changes3.6→3.9, but the requested return remains unsolved in both
seeds. This small rubric difference is not convincing evidence of improvement,
given the observed judge repeatability problems. No adaptive success is claimed.

For **both** pairs, all first12 chunks are pixel-identical and the entire action
sequence matches. Retrieval activates only on chunk12; that is also the first
raw-pixel divergence. Thus the final change is causally isolated to the memory
intervention, but it does not restore the starting view. See paired-summary.json
and return-contact.jpg, plus the complete videos. Each archive costs19.766GB
CPU RAM; observed H200 allocation peaks72.66→73.03GB on this short episode.

The archive contains latentframes6..21, excluding the exact initial neutral
view0..3. The next exploratory replay compares the existing start-view
re-anchoring implementation with fresh stock and four-view controls, using both
seeds and matched H100 workers. It changes both archive content and trigger
timing; it is not an archive-only ablation. Earlier start-anchor tests elsewhere
already had framing failures, so this is a targeted diagnostic, not a presumed
solution. Plan: reproduction/start-anchor-replay-plan.json.
