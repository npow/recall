# First complete stock episode

**Integration limitation found afterward:** the moving-camera audit detected
that v3 computed camera rays under the denoiser's BF16 autocast, whereas native
LingBot computes them in FP32. Neutral inputs did not expose this difference.
Protocol v4 corrects the precision scope. This v3 episode remains an archived
diagnostic and must not be reused as the corrected-native baseline.

GC007, LingBot-World2 14B causal-fast, seed 42, Haiku 4.5 player,
Claude Opus 5.5 judge with the pinned released PlayWorld rubric.

| Measurement | Result |
| --- | --- |
| Local visual consistency score | 3.2 / 5 |
| Trajectory gate | Partial; requested revisit not observed |
| Video duration | 25.8125 seconds, 413 frames |
| Player interactions | 27, including END |
| Generation wall time including initialization and player | 360.27 seconds |
| Hardware | One Modal H200 in nissan-pow |

The player moved forward, issued turns, then traveled along a different path.
It falsely declared success while the intended camera position remained 61.65
normalized units from its origin. Restoring the heading did not restore the
position. Visual review and the independent judge both show the camera ending
at a shopfront instead of the starting viewpoint. This episode does not test a
successful revisit and cannot establish a long-horizon memory improvement.

The released local rubric permits partial trajectories to receive visual
consistency scores; task-specific completion is excluded from that score.
Therefore **3.2 is not a navigation success score or a leaderboard overall**.
No proposed memory method has been compared yet. Keep this failure when
reporting subsequent complete episodes; it is not silently discarded.

* [Video](videos/GC007-stock-v3.mp4)
* [Contact sheet](videos/GC007-stock-v3-contact.jpg)
* [Full controls, player responses, judge evidence and source hashes](GC007-stock-v3.json)
* [H200 operator profile](lingbot-profile-summary.json)
* [Runtime optimization comparison](lingbot-host-index-comparison.json)

The profile captured two warmed chunks. Self-attention accounted for 37.6% of
exclusive GPU operator time; denoising and clean-cache transformer passes took
9.62 GPU seconds and streaming VAE decode took 1.15 GPU seconds. Scope times
are inclusive; CPU synchronization waiting overlaps GPU execution.

A diagnostic host mirror of cache indices reduced synchronization calls from
1,548 to 108 across those two chunks. All 10 generated raw chunk hashes matched
stock exactly, including cache eviction. Unprofiled warm chunks averaged
5.82 seconds stock vs 5.52 seconds with host indices: an observed 1.055x speedup.
This is one run per mode on separate H200 containers, so timing noise remains.
The optimization was not enabled for this scored stock episode. Checkpoint,
sampler, cache length, sink anchors, precision, and control mapping were unchanged.

The full CPU/CUDA Chrome trace is retained in Modal volume
`worldsync-lingbot-results/profile-h200-1794701964242320114/trace.json.gz`
and locally at `outputs/lingbot-reproduction/h200-trace.json.gz`.
