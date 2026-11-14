# WorldSync

**Current milestone: reproduce a published baseline. No hillclimbing yet.**

We have not reproduced PlayWorld's published SOTA or its Matrix-Game 3.0 baseline.
The previous ABot runs are exploratory and excluded from reproduction evidence.
The experiment CLIs now default to stock and block modified memory policies until
a verified reproduction report exists.

The first local target is **Matrix-Game 3.0: 1.14 overall**, under the original
PlayWorld protocol. SOTA references are **Genie 3: 2.12** and **LingBot-World2:
1.82**; reproducing Matrix is not a claim to reproduce those models.

Read [the reproduction protocol, audit and completion criteria](reproduction/README.md).
[Machine-readable state](reproduction/state.json) records the closed gate.

All 171 benchmark inputs are downloaded and verified against the pinned source;
the [asset inventory](reproduction/assets.json) also includes three public demo
videos. Recovering exact generation/evaluation settings remains unresolved.
The [stock Matrix adapter](reproduction/matrix-adapter.md) now preserves the
upstream generator, verifies native control tensors, and provides a resumable
171-case queue. Its reconstructed controller/settings remain explicitly labeled.
The [live workflow status and coverage](reproduction/matrix-progress.json) are
pushed as the stock suite advances. Missing cases prevent an overall score.
No replacement controller or judge will be labeled as numerical
reproduction. The earlier no-Gemini instruction remains in effect.

The [historical implementation](docs/exploratory-setup.md) and
[exploratory results](results/) are preserved for inspection. They cannot justify
starting optimization. API credentials, weights and downloaded benchmark media
remain outside Git. Tested changes and audit results are pushed to this private
repository as they are completed.
