# Local OE020 prompt ablation

One matched pair on local Thor:60chunks, seed42, canonical controller, same stock checkpoint/cache/sampler. Only generation text changes: task-only versus caption-plus-task. Source archived at315cdce before launch so collaborator edits do not change the second arm.

[Declared plan](plan.json). Live outputs: `outputs/local-caption-pair-OE020/manifest.json`, with per-arm logs and original videos. The task was selected after observing earlier H100 failures, so this is a development experiment. No result is available yet. No cloud GPU allocated.

## Reproduction

Run `python scripts/run_local_caption_pair.py` from a checkout with the local LingBot environment and weights. It refuses an existing `outputs/local-caption-pair-OE020` directory, archives tracked `worldsync` source at the launch commit, and runs the two arms sequentially with terminal return codes recorded. Shared model weights and benchmark data are referenced locally. It does not start cloud GPUs or submit judge calls. The currently active run was started with the equivalent temporary launcher before this script was added.

Before any judging, run `.venv-lingbot/bin/python scripts/verify_local_caption_pair.py --output outputs/local-caption-pair-OE020/pair-verification.json`. It rejects incomplete episodes, mismatched settings/hardware/source/camera traces, altered chunks and incorrect encoded duration/frame counts. It does not judge quality. Three CPU tests exercise real encoded frame counts, changed-source rejection, mismatched hardware and pending-run rejection.
