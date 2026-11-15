# fp8 KV cache for LingBot-World2 (`--fp8-kv`, default off, diagnostic)

> **Integration note (main).** `--fp8-kv` is wired into `worldsync/lingbot_rollout.py` (default off, mutually exclusive with the
> other memory diagnostics, native fixed paths only). Measured on the fork's rollout, not re-run on main. The 4.9 MB
> `comparison.png` image is not committed; `results.json` and `table.md` are.


Stores the rolling self-attention K/V in float8 e4m3 with one bf16 scale per (token, head); the attended window is
dequantized to bf16 right before attention, layer by layer. Weights stay bf16. v2 (current): the chunk being generated
attends to its own exact bf16 K/V, and only past frames come from the fp8 cache (`--fp8-kv-quantize-current` restores v1).
Not bit-exact with bf16. Code: `worldsync/lingbot_fp8kv.py`; tests: `tests/test_lingbot_fp8kv.py`; runs through the
persistent runner (`scripts/modal_lingbot_runner.py`, jobs `scripts/make_fp8_jobs.py`, report `scripts/fp8_report.py`).

## Memory and speed (H200, GC007/GC001, 8-16 chunk fixed paths)

| window (sink 6) | cache | peak allocated | warm s/chunk (short / 16-chunk paths) |
| --- | --- | --- | --- |
| 18 (stock, bf16) | 23 GB | 80.2 GB (earlier validation) | 5.8 |
| 46 bf16 | 59 GB | 114.8 GB | 5.9 / 7.4 |
| **46 fp8 (v2)** | 30 GB | **86.8-87.6 GB** | 6.5 / 8.1-8.3 (+8-10%) |
| **102 fp8 (v2, 32-chunk path)** | 68 GB | **126.7 GB** | 12.6 |

fp8 at 46 frames is -28 GB (-24%) against bf16 46. 102 frames in bf16 would need roughly 190 GB (does not fit an H200).
**Peak-memory caveat:** the warm runner's `peak_allocated_bytes` stays inflated after a high-memory job in the same
container: in the v1 batch every job that followed a bf16 46/6 reference reported 114.8-115.0 GB, including fp8 jobs.
Clean numbers are the first job in a container (86.8 GB) and the v2 batch, which had no bf16 job. A fresh one-shot run
per config was not done.

## Numerics (bf16 shadow cache on layers 0, 10, 20, 30, 39; GC007 turn_wait_return)

Attention-output relative error, mean (max): v2: 0.6% (0.8%), 1.1% (1.5%), 1.4% (2.2%), 1.0% (1.6%), 1.1% (5.4%).
v1 (current chunk also rounded) was 6.2%, 1.7%, 2.4%, 2.3%, 3.9% (max 10.4%). Chunk-0 PSNR vs bf16 is 35-45 dB (v2) but
falls to 10-13 dB by chunk 3: the two runs diverge to different content (diffusion sampling is chaotic), so PSNR is
not a fidelity measure past the first chunk.

## Revisit quality (DINOv2-small, final frame vs frame 12; one seed per cell; stock noise is about 0.1)

| path / scene | bf16 46 | fp8 46 v2 | fp8 46 v1 | fp8 70-102 |
| --- | --- | --- | --- | --- |
| GC007 turn_return | 0.847 | 0.814 | 0.787 | 0.787 (70) |
| GC007 turn_wait_return | 0.570 | 0.555 | 0.429 | 0.737 (70, v1), 0.736 (102, v1) |
| GC001 turn_return | 0.527 | 0.641 | 0.535 | 0.535 (70) |
| GC001 turn_wait_return | 0.807 | **0.228** | 0.428 | 0.678 (70, v1) |
| GC007 turn_long_wait_return (24-chunk hold) | 0.498 | n/a | 0.409 | 0.522 (102, v1), 0.469 (102, v2) |

Seed 43 (v1): GC001 wait 0.247, GC007 wait 0.412, turn_return 0.686/0.821. Images: `comparison.png` (frame 12 | bf16 end | fp8 end).
Honest reading: on the short paths fp8 keeps the bf16 revisit behaviour (end views match the start scene). On GC001
turn_wait_return the v2 fp8 run fails to recall the dining view that bf16 46 recalls (0.23 against 0.81), and v1 at
window 70 did recall it; with one seed per cell, fp8 sensitivity and ordinary seed noise cannot be separated. On the
24-chunk hold bf16 46 loses the start building (blue wall), and fp8 at 102 shows the facade lettering again (v1
clearly, v2 partly, a different angle), but DINO does not separate them (0.47-0.52 against 0.50).

## Not done / limitations

* No H100 run: fp8 46/6 peaks at about 87 GB, 7 GB over an 80 GB card; the text-encoder CPU offload (about 8 GB) would
  just fit but was not tested. Options (b) CPU-offloaded cold frames and (c) top-N cold tokens were not built.
* Single seed per cell for v2; no second scene for the long path; DINO is a crude proxy; no judged score.
* Not combined with the retrieval/memory flags (they read `kv_cache` and would see fp8); rollout rejects combinations.
* The `fp8-c100` job in the first batch failed validation (window minus sink must be a multiple of 4), so 102 was used.
