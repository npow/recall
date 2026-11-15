# Full-model FlashAttention4 timing pilot

**Warm generation improves from5.75 to4.95 wall seconds per simulated second on H100:1.16× throughput versus the established stack.** Same-stack Torch2.9 SDPA versus FA4 improves from5.98 to4.95 seconds:1.21× throughput.

All three runs complete the same10-chunk WAIT path (157frames,9.8125seconds), GC007, seed42, native fixed camera, stock18-frame/6-sink KV cache, original unquantized checkpoint, four denoising steps, and identical text-encoder offloading. No retrieval or player calls are used. Every chunk passes the finite-output and native-camera audits. Warm measurements are the final four full-window chunk intervals and include decoding and output transport.

| Runtime | Warm seconds / simulated second | First chunk, including loading | Complete clip wall time | Peak allocated GPU |
|---|---:|---:|---:|---:|
| Torch2.6 / CUDA12.4 SDPA | 5.748 | 114.88s | 165.46s | 71.893GB |
| Torch2.9 / CUDA12.8 SDPA | 5.978 | 96.89s | 148.46s | 71.893GB |
| Torch2.9 / CUDA12.8 FA4 | 4.949 | 145.81s | 189.34s | 71.893GB |
| Same FA4, optional failed addon disabled | 4.941 | 114.56s | 157.99s | 71.893GB |

The first FA4 import tries an optional DLPack addon, then warns that its JIT fallback failed. Disabling that optional path and retaining TVM-FFI's official older-Torch stream helper produces **all10 raw chunks pixel-identical to the original FA4 run**. Warm timing remains the same; observed startup falls by31seconds. Independent-container I/O variability remains, so this is not a controlled attribution of the entire cold-start difference.

Cold loading still dominates a short episode. The complete ten-second clip does not receive the full warm throughput gain. Reusing a loaded pipeline across trials is a promising separate optimization. Peak allocated GPU memory is unchanged in this no-retrieval pilot because another stage sets the peak. The [retrieval buffer lifetime change](../lingbot-read-cache-lifetime/) separately reduces memory for memory-enabled rollouts.

FA4 and the SDPA controls are **not pixel-identical**. All three contact sheets retain the mall roof, sign, facade and framing without obvious corruption in an unblinded inspection. Ten matched decoded end-of-chunk frames give mean DINO cosine0.99417 for FA4 versus same-stack SDPA, and0.99455 for old versus new SDPA. This stationary diagnostic does not establish moving-camera, dynamic-process or long-horizon quality equivalence. The established scored experiments keep their original Torch2.6/SDPA stack.

Evidence: [timing summary](summary.json), [frame comparison and video hashes](frame-comparison.json), [addon parity comparison](no-addon-comparison.json). Complete videos and contacts: [stock SDPA](sdpa26/rollout.mp4), [new-stack SDPA](sdpa29/rollout.mp4), [FA4](fa4/rollout.mp4). All four raw manifests and the Modal results directory identifiers are retained here. The no-addon video's raw pixels match the included FA4 clip at every chunk.

Run `modal run --profile nissan-pow scripts/modal_lingbot_runtime_benchmark.py`, then `modal run --profile nissan-pow scripts/modal_lingbot_no_dlpack.py::main`. Frame analysis uses `HF_HUB_OFFLINE=1 .venv-lingbot/bin/python scripts/analyze_lingbot_runtime.py` with the pinned DINO checkpoint cached locally.

The new backend is explicitly selected with `WORLDSYNC_LINGBOT_ATTENTION=fa4` in a compatible environment. `TVM_FFI_DISABLE_TORCH_C_DLPACK=1` selects the validated addon-free path for this pinned runtime. The default remains SDPA; there is no implicit fallback from an unavailable FA4 backend. Six CPU semantic tests cover length masking, padded queries, scaling and unsupported attention requests; the complete GPU pilots exercise the actual FA4 integration.
