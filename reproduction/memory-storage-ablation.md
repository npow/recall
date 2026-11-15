# Cheaper storage for memory frames (ablation; diagnostic, not a PlayWorld score)

Question: the bounded pose memory archives canonical bf16 K and V for every layer, about 1.24 GB per latent
frame at this resolution. How much less can be stored before the revisit effect is lost?

Setup: `--budget-memory --budget-rope native --budget-retrieved 10` on the final-run harness (director commit
`abfa73a`), with a storage codec (`--budget-store`) and a layer subset (`--budget-layers`), both default off and
allowed only on fixed native-camera paths. Scripted paths at 68 deg/s: `turn_hold_return` (turn away, hold
6 chunks, turn back; 14 chunks) and `turn_return`. Metric: DINOv2-small cosine of the final frame against
frame 12 (frame 0 is in `results.json`), plus the frame sheets, which I looked at. Lambda H100 PCIe, seed 42
unless stated. The disk-backed and latent-recompute back-ends are a sibling experiment (memory v2), not here.

Reads keep retrieved frames in stored form (GPU when under 7.5 GB for all layers, else pinned CPU) and decode
per attention call, so ten retrieved frames fit an 80 GB H100: peak 69 to 75 GB, against 93 GB on an H200
with the original implementation.

## Results

| variant | MB per archived frame | ratio | GC007 hold | GC001 hold | visibly returns (GC007 / GC001) | s | peak GB |
|---|---:|---:|---:|---:|---|---:|---:|
| stock (no memory) | 0 | n/a | 0.352 | 0.192 | no / no | 132 | 68.8 |
| exact bf16 | 1235 | 1.0x | 0.780 | 0.567 | yes / yes | 209 | 75.1 |
| fp8 (scale per token and head) | 627 | 2.0x | 0.818 | 0.619 | yes / yes | 174 | 74.4 |
| int8 (scale per token) | 618 | 2.0x | 0.789 | 0.352 | yes / no | 218 | 74.3 |
| layers: even (20) | 618 | 2.0x | 0.538 | not run | no | 169 | 74.2 |
| layers: first 20 | 618 | 2.0x | 0.574 | 0.203 | yes / no | 165 | 74.2 |
| layers: last 20 | 618 | 2.0x | 0.559 | 0.293 | no / no | 220 | 74.2 |
| layers: middle 20 | 618 | 2.0x | 0.504 | not run | no | 161 | 74.2 |
| layers: every 4th (10) | 309 | 4.0x | 0.609 | 0.331 | no / no | 148 | 71.1 |
| layers: first 10 | 309 | 4.0x | 0.493 | 0.204 | no / no | 207 | 71.1 |
| layers: last 10 | 309 | 4.0x | 0.397 | not run | no | 151 | 71.1 |
| pool 2x2 (repeated at read) | 319 | 3.9x | 0.609 | 0.361 | no / no | 166 | 71.3 |
| pool 4x4 | 85 | 14.5x | 0.404 | 0.158 | no / no | 220 | 69.0 |
| top 25% tokens by value norm | 309 | 4.0x | 0.562 | not run | yes | 153 | 70.5 |
| top 10% tokens by value norm | 124 | 10.0x | 0.608 | 0.367 | yes / no | 145 | 68.8 |
| random 25% tokens (control) | 309 | 4.0x | 0.693 | not run | partly | 204 | 70.5 |
| low rank 256 | 271 | 4.6x | 0.604 | 0.329 | yes / no | 245 | 70.9 |
| low rank 64 | 68 | 18.2x | 0.821 | 0.394 | yes / no | 175 | 68.8 |
| pool 2x2 + fp8 | 162 | 7.6x | 0.626 | not run | no | 223 | 69.8 |
| pool 2x2 + fp8, even layers | 81 | 15.2x | 0.374 | not run | no | 151 | 69.0 |

"Visibly returns" is my reading of the final frame in `memory-storage-ablation/*.jpg`: on GC007 the Primemall
facade with the roofline (exact returns to it from a closer viewpoint, with an umbrella in front, not to the
identical frame-12 view); on GC001 the dining room with the chandelier.

## What it shows

- **The exact baseline reproduces on this hardware**: 0.780 against 0.352 for stock on GC007 (0.77 and 0.30 on
  Modal), 0.567 against 0.192 on GC001.
- **8-bit, 2x**: fp8 keeps the revisit on both scenes (0.818 and 0.619). int8 with one scale per token keeps it
  on GC007 (0.789) and loses it on GC001 (0.352). bf16 to 8 bit is 2x, not 4x.
- **Layer subsets do not work.** Every subset loses most of the effect on at least one scene. No half or
  quarter of the layers carries the memory; substituting memory in some layers and the stock window in others
  seems to leave the model with inconsistent context. The last ten layers alone are the worst.
- **Spatial pooling does not work.** 2x2 pooling already lands somewhere else; 4x4 is at stock level.
- **Token subsets are mixed.** Ranking tokens by value norm is no better than a random subset on the metric
  (0.562 against 0.693 at 25%). Visually the top-25% and top-10% runs do return to the facade on GC007, and
  top-10% does not return on GC001.
- **Low rank works on one scene only.** On GC007, rank 64 stores 68 MB per frame (18x less) and returns to
  the facade (0.821); rank 256 also returns, with a lower metric value (0.604). On GC001 neither returns to
  the dining view (0.394 and 0.329; the final frames show a sofa and a lamp). So low rank is not a safe
  replacement either.
- **Second seed (GC007 hold, seed 43):** stock 0.718, exact 0.718, fp8 0.755. The metric does not separate
  them, but the frames do: exact and fp8 end on the Primemall facade, stock ends on a wide plaza with a
  different building. The fp8 result therefore holds on a second seed by eye, and the DINO number alone
  would have hidden the memory effect entirely on this seed.

## Ranking and recommendation

1. **fp8** is the only variant that kept the revisit everywhere it was tested (both scenes on seed 42, GC007 on
   seed 43): half the storage.
2. **Rank-64 low rank** (18x) and **top-10% tokens** (10x) keep the revisit on GC007 and lose it on GC001.
3. int8 with a per-token scale, pooling and layer subsets should not replace exact storage.

For memory v2: fp8 can replace exact K/V in the archive. Nothing smaller than half is reliable in these
tests, so a large reduction has to come from a different design (storing the frame latent and recomputing
K/V, or exact K/V on disk), not from compressing K/V.

## Limits

- One seed per cell, except stock, exact and fp8 on GC007 (two seeds). Stock scored 0.352 on seed 42 and
  0.718 on seed 43, so the metric moves more between seeds than between most variants.
- DINO of a single final frame is a crude proxy; the frame sheets disagree with it in several rows (noted
  above), and the "visibly returns" column is one reader's judgement of one frame.
- Two scenes, scripted rotation-only paths with a 4.5 s hold. Nothing here is a judged episode.
- Not run by the cut-off: int8 on seed 43, all seed-43 jobs on GC001, and `turn_return` (the Lambda instances
  were reassigned). GC001 has 13 of 20 variants.
- The low-rank store uses a randomised factorisation (`torch.svd_lowrank`, 2 power iterations); its first
  version crashed under bf16 autocast and was fixed before these runs.
- Selection, eviction (24 frames) and the attended window are those of the bounded pose memory. Its known
  problems on real episodes (commanded-pose retrieval, short archive) are unchanged by the storage format.

Files: `memory-storage-ablation/table.md`, `results.json`, frame sheets per scene; job lists
`memory-storage-jobs*.jsonl`; code `worldsync/lingbot_memstore.py`, `StoredBudgetMemory` in
`worldsync/lingbot_budgetmem.py`; tests `tests/test_lingbot_memstore.py`.
