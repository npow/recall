# Local Thor memory feasibility and isolated prototypes

No full14B local model was launched for this assessment. The observed machine has131.86GB physical RAM visible to Linux (122.8GiB), with108.74GB available at the recorded snapshot. Thor shares this RAM with CPU processes; GPU allocations are not an independent memory pool.

The existing BF16 CPU archive occupies19.766GB. Symmetric int8 codes with one FP32 scale per token/head vector reduce it to10.192GB, including scales:48.44% less storage. A CPU probe on BF16 features gives relative reconstruction error0.00647. This is lossy feature compression, not a validated video-quality improvement.

An isolated `Int8ArchiveMemory` subclass stores only compressed CPU features and dequantizes selected frames on the GPU before the existing BF16 read cache and RoPE operation. Native persistent KV tensors and model parameters stay unquantized. Six CPU tests cover error bounds, storage accounting, nonfinite rejection, encoder disposal and a tiny real upstream attention block. In that block native KV tensors and pre-retrieval outputs remain exact; retrieved attention output changes by less than3% in the tested case. Longer generated-video quality remains untested.

The separate one-shot text-encoder proxy computes the original context, then replaces T5 parameters with meta tensors immediately after encoding, before native VAE/KV allocation. It preserves the frozen rollout's later `.cpu()` call as a no-op and rejects another encode. A CPU test checks exact encoded output and removal of the parameter payload. It is deliberately unsuitable for persistent multi-request workers.

Using the existing72.76GB GPU peak as a conservative proxy,10.19GB compressed archive,5GB process/transient allowance and6GB allocator allowance gives93.95GB for the application. That leaves14.79GB available at the snapshot, below the required16GiB (17.18GB) system reserve. **The conservative full-memory budget does not pass.** T5 disposal should improve the actual stage peak, but that must be measured on Thor rather than assumed from a discrete GPU.

The proposed next step is a separately guarded two-chunk, no-archive smoke test with one-shot T5 disposal. Its conservative estimate is83.76GB and leaves about25GB, clearing the16GiB reserve at this snapshot. Recheck available memory immediately before launch, monitor MemAvailable/process RSS/CUDA allocated and reserved throughout loading and execution, and terminate the isolated process if the system reserve is crossed. Only then reassess a complete baseline/quantized pair.

Data: [feasibility snapshot and CPU probe](feasibility.json). Code is isolated in `worldsync/lingbot_local_memory.py`; no normal rollout flag or default changes, and the frozen retrieval, rollout and player files are untouched. No model training is performed.
