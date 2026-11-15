# WorldSync

**Current milestone: validate archived-view memory beyond the rolling cache.**

[Historical-view retrieval](reproduction/memory-validation/README.md) now
preserves returning landmarks in a new-seed trial, an image-held-out fountain
scene, and a31.8-second mall return where stock forgets the starting building.
The same18-frame attention budget is retained, with a bounded19.765GB CPU
archive and about6% higher GPU peak allocation. Exact heading recovery and
some texture/sign drift remain unresolved; endpoint similarity can decline
when the camera overshoots. No training was used for these tests.

[Task-matched released-rubric comparisons](reproduction/task-memory-benchmark-plan.json)
are in progress. These results do not yet establish a PlayWorld score gain.

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
