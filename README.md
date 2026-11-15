# WorldSync

[Experiment leaderboard](LEADERBOARD.md) · [Method overview](METHOD.md) · [Detailed memory methods](docs/METHOD.md) · [Machine-readable results](leaderboard.json)

**Current milestone: validate archived-view memory beyond the rolling cache.**

[Historical-view retrieval](reproduction/memory-validation/README.md) now
preserves returning landmarks in a new-seed trial, an image-held-out fountain
scene, and a 31.8-second mall return where stock forgets the starting building.
The same 18-frame attention budget is retained, with a bounded 19.765GB CPU
archive and about 6% higher GPU peak allocation. Exact heading recovery and
some texture/sign drift remain unresolved; endpoint similarity can decline
when the camera overshoots. No training was used for these tests.

[The audited local task pilot](reproduction/task-memory-benchmark/README.md)
improves the same Claude rubric from **2.47 to 3.57** across three valid
scripted tasks. One released image/caption mismatch is quarantined and retained
explicitly. This is a local subset result, not a published PlayWorld score.

[The first adaptive four-view pilot](reproduction/adaptive-four-view/README.md)
remains unsuccessful: local rubric means3.6→3.9, but neither seed visually
returns to the initial view. All first12 chunks match stock exactly; retrieval
activates only on the final return chunk. The small score gain is inconclusive.

[Preserving recent context](reproduction/protected-recent-memory/summary.json)
lets eight historical views run without the earlier resets on the development
scene. Blinded comparisons weakly prefer this22-frame variant; fresh-image and
new-seed checks also weakly prefer it, but task confirmation below does not. The validated four-view method remains the default.

[Streamed player overlap](reproduction/player-streaming/README.md) hid27.7seconds
of response explanations in a live run, with all10 generated chunks matching
the previous stock prefix. The player stopped earlier, so this is not a matched
full-episode speedup. Full-response validation and END checks are preserved.

[Temporary-buffer release](reproduction/lingbot-read-cache-lifetime/README.md)
reduces peak GPU allocation by1.59GB with all32chunks pixel-identical.
[FlashAttention4 measurements](reproduction/lingbot-attention-optimization.md)
show about1.9x faster attention and16% faster warm stock generation onH100.
[Pipeline reuse](reproduction/lingbot-batch-validation/) preserves all42 raw chunk
hashes across three moving-camera scenes and removes repeated model loading.

[Task confirmation](reproduction/task-context-validation/summary.json) rejects
promoting eight-view retrieval: fresh three-task means are3.95 for four views
and3.63 for eight. Separately normalized attention gates also fail full-video
review. The four-view method remains selected.

[Judge repeatability](reproduction/judge-repeatability.json) is limited: the
same beach video and identical prompts scored2.7 and3.75 in separate calls.
Treat small differences as uncertain and inspect complete videos.

[Separate local geometry maps](reproduction/spatial-local-submaps/summary.json)
pass more revisit localization checks than a fused map on two mall seeds, but
still fail the bridge returns and the drifted stock scene. These sparse maps
are diagnostics; they have not been used to condition generation.

[The four/eight-view ablation](reproduction/multiview-retrieval/README.md) finds
that eight historical views degrade continuity when they displace recent
context. Four diverse views tie the four-nearest reference.

Native moving-camera input parity is verified. Complete local stock episodes
show substantial seed and player variance; see the [five-seed diagnostic](reproduction/GC007-v5-seeds.json).
We have not reproduced the published overall score. Adaptive tuning remains
gated; explicitly declared native fixed-path experiments run separately.

The [full-history pilot](reproduction/context34-v1/README.md) retains 34 instead
of 18 latent frames. On one turn-and-return scene it recovers the starting
building better: endpoint DINO similarity rises from 0.579 to 0.848 and the
blinded full-video review prefers it. It costs 25% more peak GPU memory and
about 8% more warmed generation time in this pair. This is a short diagnostic,
not a PlayWorld score. [Frozen additional seed/scene comparisons](reproduction/context-validation-plan.json)
test whether it generalizes. The [earlier retrieval trial](reproduction/roundtrip-v4/README.md)
failed and remains published.

The current model is **LingBot-World2 14B causal-fast**. Its hosted service scored
**1.82** in the published PlayWorld table; that does not establish a score for
the public local checkpoint. The closed reference is **Genie 3: 2.12**.
We are using the official **14B causal-fast** release, without quantization or
substitution of the smaller 1.3B model.

Read [the reproduction protocol, audit and completion criteria](reproduction/README.md).
[Machine-readable state](reproduction/state.json) records the closed gate.

All 171 benchmark inputs are downloaded and verified against the pinned source;
the [asset inventory](reproduction/assets.json) also includes three public demo
videos. Recovering exact generation/evaluation settings remains unresolved.
The [LingBot setup and audit](reproduction/lingbot.md) records the pinned source,
weights and remaining protocol differences. [Live status](reproduction/lingbot-progress.json)
is pushed as work advances. Missing cases prevent an overall score.

Matrix generation is stopped following the target change. Its 57-frame native
smoke test passed; the full benchmark was cancelled before completing a case.
The [Matrix adapter and audit](reproduction/matrix-adapter.md) are archived for
inspection and do not count as a reproduced score.
No replacement controller or judge will be labeled as numerical
reproduction. The earlier no-Gemini instruction remains in effect.

The [historical implementation](docs/exploratory-setup.md) and
[exploratory ABot results](results/) are preserved for inspection and excluded
from LingBot reproduction evidence. API credentials, weights and downloaded benchmark media
remain outside Git. Tested changes and audit results are pushed to this private
repository as they are completed.
