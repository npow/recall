> Historical audit. Its published-score agreement requirement was superseded by the user's local-baseline instruction. See [current policy](../reproduction/README.md).

# Reproduction before optimization

**Status: not reproduced. Hillclimbing is locked.**

The ABot experiments in `results/2026-11-14-*` are exploratory engineering checks.
They changed the model, controller, judge and budget, and provide no evidence of
reproducing the published PlayWorld results. They are excluded from this milestone.

The current generation target is **LingBot-World2**, the published open-model SOTA
at 1.82. The closed reference is Genie 3 at 2.12. The earlier Matrix target (1.14)
was cancelled at the user's instruction after its native smoke test succeeded.

## Frozen reference

| Item | Published protocol / current evidence |
| --- | --- |
| Cases | 171: GC 48, IF 50, insight 30, out-of-sight 43 |
| Player | Claude Haiku 4.5, not our GPT-6.1 controller |
| Control | Human reference plan plus adaptive Keep/Stop/Extend/Correct/End decisions |
| Budget | 40 interactions; reproduce the actual per-model action timing |
| Judge | Gemini 3.1 Pro; Claude replacement is not numerical reproduction |
| Sampling | Primary 10 FPS, 384×216 cells, 5×5; detail 0.5 FPS, 800×450, 2×2 |
| Aggregation | Equal mean of the four dimension means, not one mean over all cases |
| LingBot-World2 scores | GC 2.04, IF 2.13, insight 1.95, out-of-sight 1.16; overall 1.82 |
| Weights and settings | Exact paper checkpoint revision, sampler/decoder/precision and adapter settings are not yet verified |

Sources: [paper, Appendix A](https://arxiv.org/html/2608.13552v2#A1),
[released code](https://github.com/hku-sail/PlayWorld),
[dataset](https://huggingface.co/datasets/jocelynd/playworld-bench),
[project examples](https://kxding.github.io/project/PlayWorld/).

## Audit findings to resolve

* The paper evaluates LingBot-World2 through its web interface. Local inference
  uses the released 14B causal-fast checkpoint; exact equivalence to the hosted
  benchmark variant and settings still requires verification. The public player
  release has no LingBot-specific adapter. This difference must remain visible.
* The earlier Matrix audit found a SANA adapter but no Matrix-specific adapter.
  The authors' [local-adapter issue](https://github.com/hku-sail/PlayWorld/issues/3)
  points to SANA as an example. Its model-specific action grouping is not evidence
  of Matrix's original settings. We will not guess them and call that reproduction.
* The paper describes weighted binary VQA answers mapped by `1 + 4R`. The released
  scorer requests 1–5 category scores with additional policies. Establish which
  scorer/data revision produced the reported table before using it as acceptance
  evidence. Our previous Claude adapter must not silently stand in for that audit.
* The paper and released README describe reference-plan handling differently.
  Match the executed controller and prompts, not just the model name.
* The public project has selected demonstration videos, including Genie GC004
  and Matrix GC007/OE001. They are useful reference artifacts, not the complete
  171-case rollout archive or original per-case judge cache. Scoring a few selected
  demos does not reproduce the aggregate SOTA result.
* Gemini use remains unauthorized under the user's earlier instruction. No Gemini
  calls will be made. Exact score reproduction needs either the original saved
  judgments or explicit resolution of that constraint. A Claude-calibrated study
  would be a different protocol and cannot unlock this milestone as-is.

## Prepared assets

The [stock Matrix adapter reconstruction](../reproduction/matrix-adapter.md) records recovered
native timing, source and checkpoint history, exact control-tensor checks, runtime
commands, and remaining deviations. It restores the missing model integration
without modifying the generator or claiming that unknown settings are verified.

[assets.json](../reproduction/assets.json) inventories all 171 cases and their input images at
dataset revision `697e177194a1c4d8afe9068756fe743ac23ee1e8`.
Every metadata file and image was verified against the immutable upstream Git
blob hash or LFS SHA-256, and every image was checked for decoding errors.
The inventory also records hashes and video metadata for three selected public
demonstrations. None of these assets constitute a reproduced score.

Rebuild the inventory without making generation or judge API calls:

```bash
.venv/bin/python scripts/prepare_reproduction.py --with-reference-videos
```

This downloads benchmark media into ignored `external/` directories. Omitting
`--with-reference-videos` prepares only the complete benchmark inputs.

## Completion criteria

1. Pin the dataset, unmodified generation source and exact paper checkpoint.
2. Recover and freeze the LingBot controller, action mapping, budgets, preprocessing,
   sampler, decoder and precision settings. Record any hardware substitutions.
3. Verify scoring and aggregation against the exact released/paper version and
   reference artifacts. Retain missing data as missing; do not silently exclude it.
4. Generate the full unmodified baseline, preserving every case ID, action trace,
   video, termination reason and failure. Report all four dimensions and coverage.
5. Establish score agreement with the published reference and explain residual
   variation. Fix numerical acceptance criteria before testing modifications.
6. Save a verified reproduction report. Only then change `state.json` to the
   hillclimbing phase and run one modification at a time under the frozen protocol.

There is no verified report yet. The command-line experiment runners default to
stock and reject non-stock modes while this gate is closed. Merely running a
stock model, passing unit tests, downloading a video, or copying the published
table does not satisfy reproduction.

Historical setup and implementation details remain in
[the exploratory documentation](../docs/exploratory-setup.md).
