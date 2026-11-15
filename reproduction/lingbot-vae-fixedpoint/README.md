The standalone actual Wan2.1 VAE produces exactly identical normalized latent bytes for GC027 at205 and637RGB frames. Complete encoder cache state (24tensors plus2unused entries) becomes bitwise fixed after30 encoder calls; repeated positive-zero inputs therefore reuse that encoder output. The original full52/160-group conv1 and normalization are preserved. All VAE parameter version counters remain unchanged. No default or cloud generator is modified.

| RGB frames | Native encode | Prototype encode | Speedup |
|---|---:|---:|---:|
|205|34.29s|21.83s|1.57x|
|637|107.26s|21.60s|4.97x|

This is a local Thor/Torch2.15 result, not an H200/Torch2.6 measurement. The full normalized output hashes match byte for byte. The prototype checks every cached tensor, its metadata and the encoder index; it does not infer convergence from only the last latent. It takes independent cache snapshots so even in-place changes are detected. CPU tests cover the real tiny causal VAE at13/205/637frames, nonzero-tail fallback and a constant-output/changing-cache adversary. They pass5tests.

The initial eligibility check uses byte-wise count_nonzero, which materializes a large reduction temporary. At637frames prototype CUDA allocation peaks at30.00GB/reserved41.38GB versus10.24GB native. This is a defect to fix before adoption. The original supervisor application estimate20GB was low; its physical-memory guard still preserved16GiB, with measured minimum71.21GB available. Future standalone checks conservatively budget55GB plus16GiB until the improved reduction is profiled. The original measured artifacts and code version are retained.

The local GPU was released immediately after proof for the separate learned-reader experiment. No extra GPU rerun or cloud request was started. A lower-memory byte-wise any eligibility check is a CPU-only follow-up until hardware is free and a new measurement is authorized. Full episode raw-pixel parity on the production stack remains necessary before enabling this optimization there.
