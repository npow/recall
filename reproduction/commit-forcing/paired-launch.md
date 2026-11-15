The frozen GC010 pair uses every action from the existing fourteen-chunk reference at45degrees/second, not the directory label as a per-second rate. Both fresh H200 arms use the same source/checkpoint, native18/6 cache, Torch2.6/cu124, SDPA, shift10/four ODE steps, caption and seed42. No retrieval, reader, LoRA, frame shortening or extra denoising forward is enabled.

One pipeline is loaded once. The stock arm uses an explicitly tested read-only observer; the second enables Commit Forcing. Every arm restores native forward bindings, model hooks and scheduler binding before the pipeline reuse audit. Actual initial noise blocks and all three scheduler re-noise draws per chunk are hashed. The first three clean K/V entries from every native layer are retained on GPU and compared byte for byte; this validation adds14.82GB to measured peaks. Reported timings are not an uninstrumented performance benchmark.

The hard gates require all fourteen chunks/native camera audits, first three raw rendered chunks identical, first clean-cache change on chunk2, matching actual noise draws, correct read lengths and unchanged cache indices/weights. The old H100 stock output is a diagnostic reference; fresh H200 stock is the control. Its old missing image/caption hashes are not invented. Current inputs and all relevant code files are pinned by SHA256; drift fails preflight. Every attempt needs a fresh output directory and preserves errors plus both runs in the results volume.

Prepared, not launched:

```sh
.venv-modal-new/bin/modal run --profile nissan-pow --detach scripts/modal_lingbot_commit_pair.py \
  --plan reproduction/commit-forcing/GC010-seed42-h200-plan-v1.json \
  --output outputs/lingbot-reproduction/commit-pair-GC010-v1
```

The launcher writes attempt/result manifests and downloads complete videos. All intermediate chunks, complete read traces and stdout remain in the named Modal result directory. If source files changed after freezing, preserve the failed attempt and explicitly freeze a new plan version before retrying. This paired fixed-control diagnostic is not a leaderboard result.
