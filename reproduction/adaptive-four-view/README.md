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
Candidate results are pending; no adaptive improvement is claimed.
