# LingBot-World2 reproduction target

The user switched the starting model to the strongest open row of the published
PlayWorld table: **LingBot-World2, overall 1.82**. Dimension references are GC 2.04,
IF 2.13, insight 1.95, and out-of-sight 1.16. Hillclimbing remains locked.

* Source: [Robbyant/lingbot-world-v2](https://github.com/Robbyant/lingbot-world-v2),
  revision `1895d300d8ac936401689b26389f51cbd36530eb`.
* Weights: [official 14B causal-fast](https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-fast),
  revision `5c33dd40b213598c418fd25bff30fdbd23fd38a7`.
* The official package totals 86.07 GB including shared assets. The DiT stores
  74.18 GB of float32 weights. Inference uses the source's bfloat16 dtype; no
  third-party quantization or smaller-model substitution is used.
* The T5 and tokenizer assets are reused from the previous downloads only after
  exact upstream checksum equality. Every downloaded file is verified before use.
* Native demo settings are 480×832 maximum area, chunk size 4 latent frames,
  local attention 18 and sink 6, 16 FPS. Exact hosted PlayWorld settings are not
  documented in the local release and remain unverified.

Prepare assets:

```bash
HF_XET_HIGH_PERFORMANCE=1 .venv-lingbot/bin/python scripts/prepare_lingbot.py \
  --workers 8 --range-workers 16
```

The isolated Python 3.12 environment pins Transformers 4.51.3 and NumPy 1.26.4
to satisfy this release's requirements and reuses the host's working CUDA torch.
The local Thor port is separate from any memory or model improvement. First
validate native generation, then the adaptive benchmark integration, then the
complete stock baseline. No numerical score is reproduced yet.

The paper's LingBot-World2 row used its web interface. The released causal-fast
model is our local starting point, and matching the hosted variant, controller,
timing and evaluator remains part of reproduction. Claude diagnostic judgments
will not be presented as the original Gemini judgments. No Gemini calls are made.

The downloader uses eight concurrent file transfers. Fresh large shards use the
native HF Xet client in high-performance mode with adaptive internal concurrency;
existing HTTP range chunks resume with sixteen concurrent streams per file.
Each complete file is checked against its immutable upstream checksum. An initial
20-second observation measured 31 MB/s of incoming traffic on the active network
interface; this is a point measurement, not a guaranteed sustained rate or ETA.

## Native generation and adapter checks

The official root config and checkpoint index match the causal-fast model's
1,421 tensor names with no missing or extra keys. The root config is exposed
alongside the shards in a separate symlink-based runtime layout, allowing the
official low-memory loader to cast float32 storage to native bfloat16 without
duplicating the full checkpoint in anonymous host memory. Actual loading rejects
any missing, extra or mismatched weights. A small loader fixture verifies exact
BF16 casts; it is not used as a benchmark model.

The native VAE was tested on CUDA: decoding eight latent frames at once versus
two four-latent observation chunks produced identical 29-frame output (13 then
16), with maximum error zero. See [the scoped parity report](lingbot-vae-verification.json).

`scripts/run_lingbot_reproduction.py` waits for the verified full checkpoint,
runs a neutral native 13-frame smoke test, then runs the callback adapter with
the same image, seed, controls and horizon. The two raw pixel arrays must match
exactly before the full suite starts. A failure stops the workflow and retains
its artifacts; it does not trigger model/parameter search.

The adaptive adapter supplies external camera inputs through root-model hooks
and decodes observations after the native clean-cache pass. Native sampler,
absolute temporal positions, causal KV updates, sink and rolling-window policy
are unchanged. Camera mapping is explicitly reconstructed: WASD is local
translation normalized by the native helper, arrow keys turn at 60 degrees/s,
and WAIT preserves pose. The local release exposes camera poses, and its example
pipeline leaves `wasd_action=None`; equivalence to the hosted character-control
interface is unverified. This is part of the protocol gap, not a verified paper
configuration.

Moving cases allow 40 Haiku decisions with one native chunk per action. Stationary
insight cases hold WAIT through 60 chunks (59.8125 seconds) without agent calls.
All native controls and actual agent-call counts are recorded separately. These
are fixed reconstruction choices. Local baseline verification also requires
complete scoring and realized-control review; agreement with the hosted 1.82 is
not required.

The suite runner is:

```bash
.venv-lingbot/bin/python -m worldsync.evaluate --model lingbot --judge \
  --output outputs/lingbot-reproduction/suite --publish-status
```

It grades with Claude and released PlayWorld rubrics as diagnostics, retains all
case IDs and failures, and withholds an overall until all 171 cases are covered.
The durable workflow launches it only after native/adapter pixel parity passes.

The Matrix workflow has been stopped. Its successful smoke produced 57 frames
at 704×1280, with all weights and source pinned; it did not reproduce a published
benchmark result. Existing Matrix code and failed/successful attempts are retained.

The [predeclared episode split](episode-split.json) contains 126 development and
45 held-out cases, with a 16-episode tuning cohort. Identical initial-image hashes
are grouped; prior calibration cases remain in development. Benchmark assets are
not permitted as fine-tuning training examples.
