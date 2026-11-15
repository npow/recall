# In-sight failure audit

Saved-video analysis only; no new GPU generation or judge calls.

## Pixel-change diagnostic

Mean absolute difference between frames sampled one second apart, resized to160×90. Units are intensity levels on0–255. This measures pixel change, not correctness, physical progress, or camera motion. Full sample series and original video hashes: [results.json](results.json).

| Case | Legacy | Corrected |
|---|---:|---:|
| OE066 | 19.259 | 36.652 |
| OE009 | 58.140 | 10.579 |
| OE020 | 17.223 | 3.085 |

## What the evidence supports

[OE020 contact sheet](OE020-contact.jpg) samples both originals at0,10,30,59seconds. The legacy scene shows traffic and a cyclist crossing the view. Corrected pedestrians and vehicles also change, so the first judge repetition’s description of an entirely static duplicated frame is too strong. Corrected mean pixel change is much lower, but that alone does not prove missing physical evolution. Preserve the released score; do not silently substitute this diagnostic for it.

OE066 has *more* pixel change in the corrected video despite a lower score. The judge reports foreground object morphing. More motion is therefore not a useful objective by itself.

## Next controlled experiment

The existing comparison changes controller and prompt together. For in-sight tasks both arms already last59.8125seconds and make zero player calls; this removes the major early-END difference seen in moving tasks. Legacy uses a scene caption; corrected uses the task instruction. This makes prompt conditioning the first inexpensive hypothesis to isolate, not an established cause.

Run OE020 and OE066 with the frozen corrected controller, same seed42, same checkpoint, camera schedule, context budget and60chunks. Change only generation text from task-only to caption plus task. Keep the completed task-only videos as controls if the same hardware/runtime/source can be matched; otherwise regenerate both arms on the same worker. Score every full video with three fixed judge repetitions and record failures. Inspect whether traffic progresses and whether the piping tool retains identity; do not select by pixel-change magnitude.

This would require2new episodes if matching controls are possible,4otherwise. The OE020 pair has since completed on Thor; OE066 has not been launched. See the [matched local result](../local-caption-pair/README.md):1.33→1.87 on one case, with process/identity failures remaining. It is a development ablation, not evidence of a full-benchmark gain. The hackathon IF demonstration remains unchanged.

## Experimental entrypoint (CPU-tested; full OE020 pair GPU-validated)

The stock baseline still rejects a caption override. A separate experimental profile permits only in-sight caption-plus-task text with the same stock cache and canonical controller. To write a CPU preflight manifest without loading a GPU:

```bash
.venv-lingbot/bin/python -m worldsync.lingbot_caption_ablation --task-id OE020 --seed 42 --chunks 60 --output outputs/caption-OE020-preflight
```

One experimental episode, when compute is available:

```bash
.venv-lingbot/bin/python -m worldsync.lingbot_caption_ablation --task-id OE020 --seed 42 --chunks 60 --generate --output outputs/caption-OE020-seed42
```

Use OE066 for the second case. Output directories must be new. For a newly matched task-only control, use `worldsync.lingbot_released_baseline --split insight --task-id OE020 --seed 42 --chunks 60 --generate --validate-controller --output outputs/task-only-OE020-seed42`. Freeze this source and use the same runtime/GPU for both arms; record source hashes, hardware and complete outputs. The runner changes only text and the explicit experimental label relative to the canonical argument vector. Ten CPU preflight/controller tests pass. A [two-chunk Thor runtime smoke](../caption-runtime-smoke/README.md) at `ad352d1` now verifies actual model execution. A [full matched OE020 pair](../local-caption-pair/README.md) at315cdce subsequently passed GPU execution and scored1.33→1.87. This single development case does not establish broader quality improvement.
