[Commit Forcing, Section5.1](https://arxiv.org/html/2611.00937v1#S5.SS1) retains the current block plus one stored predecessor during committing, while keeping denoising history unchanged. Its main experiments use three-latent blocks; this is an unvalidated transfer to LingBot's four-latent chunks.

The isolated helper keeps native forward bytecode and changes only the attention lookup in each attached instance's private function namespace. It never patches a global function/class, reallocates the native cache, changes RoPE positions, deletes historical entries, or adds a DiT forward. Clean passes read up to eight latent frames; the first block reads only itself. Denoising reads every original native key. Existing memory wrappers are rejected; later global-attention or learned-reader changes fail before the next forward.

Seven CPU tests use the actual native attention implementation and two stacked layers in FP32/BF16. They verify first-two-block equality, later-layer commit changes, preserved older cache entries, current-token inclusion, full denoising reads through cache eviction, mixed-noise handling, unchanged parameter versions, exact native bytecode, instance isolation and cleanup after failures. The matched H200 GPU diagnostic is now complete; see the results below. No score improvement, complete paper reproduction or production default is claimed.

```sh
.venv-lingbot/bin/python -m pytest tests/test_lingbot_commit_forcing.py -q
```

## Matched H200 diagnostic

[Full evidence](GC010-seed42-h200/result.json), [compact summary](GC010-seed42-h200/summary.json), and [sampled frames](GC010-seed42-h200/contact.jpg) are retained for the frozen GC010 seed42 scripted pair. One reused pipeline completed both 14-chunk videos in260.70s. All matching gates passed: identical initial and re-noising draws, cache indices and first three rendered chunks; first clean-cache difference at chunk2 and video difference at chunk3. Denoising reads remained native. The14.82GB cache snapshot is validation overhead, not production memory.

The sampled final views still contain invented buildings instead of the initial street. This diagnostic does **not** establish a successful revisit or quality improvement. Neither arm has a Gemini score; scripted diagnostics are not the full PlayWorld benchmark. The older stock video came from different hardware and its pixel hashes differ, so only the newly matched H200 stock arm is the parity comparator.
