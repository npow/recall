# Streaming player overlap

The existing reference-guided Haiku controller is unchanged. A client proxy
streams its original request, releases only a complete valid non-END action,
and reads the remaining explanation while native generation/decode proceeds.
The complete original response, usage and explanation are restored before the
next decision. END still waits for the original challenge/validation. A truncated
stream or a changed final action fails the episode rather than receiving credit.

Five CPU tests cover partial strings, truncated replies, changed actions, and
exact preservation of both original request bodies during a premature-END retry.
The flag is opt-in: --stream-player-actions. Model weights, sampler, frames,
controller prompts, tool schema, model and token budget are unchanged.

Live H200/Torch2.6 pilot: all10 generated raw chunks and actions match the prefix
of the previous13-chunk stock episode. This proves output preservation on that
executed path, not identical stochastic player termination. The player ended at
270commandeddegrees this time; the previous episode ended at360. Both have false
completion claims, so comparing92.8seconds with244.4seconds as a speedup is invalid.

Across nine actual moving chunks, streamed action availability totaled13.28s
versus40.99s for the complete replies. Generation hid27.71s of trailing response
time, with essentially no wait for the final explanation after decoding. Using
those same live response and rendering times, the serial counterfactual is89.89s
versus62.18s streamed (estimated1.45x step throughput). That is an observational
estimate, not an independently timed matched full-episode benchmark.

The API reports10,000requests/min,10Minputtokens/min and2Moutputtokens/min;
the measured eight-worker forecast is comfortably below these limits.

Artifacts: live-preflight.json, pilot-result.json, summary.json. The video stays
in Modal results under the result_directory recorded in pilot-result.json.
