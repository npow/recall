# WorldSync

**Current milestone: verify the persistent adapter and measure a local stock baseline.**

We have not measured the complete LingBot local baseline yet.
The previous ABot runs are exploratory and excluded from reproduction evidence.
The experiment CLIs now default to stock and block modified memory policies until
a verified reproduction report exists.

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
[exploratory results](results/) are preserved for inspection. They cannot justify
starting optimization. API credentials, weights and downloaded benchmark media
remain outside Git. Tested changes and audit results are pushed to this private
repository as they are completed.
