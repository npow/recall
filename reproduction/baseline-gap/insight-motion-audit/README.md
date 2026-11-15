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

This would require2new episodes if matching controls are possible,4otherwise. It has not been launched. It is a development ablation, not evidence of a full-benchmark gain. The hackathon IF demonstration remains unchanged.
