# Local OE020 prompt ablation — complete

**Task-only1.33 → caption-plus-task1.87 (+0.53) on one selected development case.** This is a partial improvement, not a benchmark result.

| Generation text | Three fixed judge scores | Mean |
|---|---|---:|
| Task only |1.0,2.0,1.0|1.3333|
| Scene caption + task |1.6,2.2,1.8|1.8667|

[Full side-by-side at normal speed](side-by-side.mp4) · [Contact sheet](contact.jpg) · [Exact results and original hashes](comparison.json) · [Pair verification](pair-verification.json)

Same local NVIDIA Thor, Torch2.15.0+cu132, frozen LingBot-World2 checkpoint, seed42, canonical controller, camera command schedule,18/6cache and sampler. Both original videos contain957frames at16fps (59.8125seconds); neither makes a player call. The only inference-setting change is task-only text versus image caption followed by the task. Source was archived at315cdce before either arm started. Both runs exited0 and passed source/settings/chunk/video checks. The preview is resized and re-encoded for viewing; judgments used original full videos.

The judges describe frozen actors and abrupt appearances in the task-only video. Caption conditioning produces more sustained traffic/cyclist motion, but cyclist bodies deform and merge. More visible activity does not mean a coherent world. The mean gain is descriptive: three judgments of the same video are not three independent generation trials. There is no population confidence interval or full-suite improvement claim.

OE020 was selected after observing failures in the earlier H100 study. This tests a hypothesis on a development case, not held-out generalization. It does not explain all differences from the older legacy controller or reproduce hosted LingBot/Genie performance. A caption-only arm and more cases/seeds would be needed to distinguish additional effects; neither has been launched.

## Evidence and reproduction

[Declared plan](plan.json). Full originals, logs and generation manifest are retained under `outputs/local-caption-pair-OE020/`. The two original run/protocol manifests and all six judge reports are included here. Judge source was loaded from the same315cdce archive; there were no failed reports or replacement repetitions.

Run `python scripts/run_local_caption_pair.py` with the local environment and weights to create a new pair. It refuses an existing output directory, archives tracked source, and runs both arms sequentially on one GPU. It does not allocate cloud GPUs or submit judging requests.

Before scoring, `.venv-lingbot/bin/python scripts/verify_local_caption_pair.py --output outputs/local-caption-pair-OE020/pair-verification.json` checks full duration/frame counts, shared source/settings/hardware, camera commands, and chunk hashes. `scripts/finish_local_caption_pair.py` waits for completion and verification before submitting the three predeclared repetitions per video. Exclusive attempt markers prevent silent resubmission. All local generation and judging for this pair have now finished; no follow-on GPU jobs were launched.
