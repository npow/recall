# Mixed-scene, moving-camera persistent batch

**All42 raw chunk hashes match three independent fresh controls exactly.** One loaded pipeline renders GC010, GC033 and GC040 serially, with different images, captions and turn directions. Each request has14 native-camera chunks at45degrees/second, stock18/sink6 and four-view retrieval, using the established unquantized Torch2.6/CUDA12.4 SDPA stack.

| Request order | Fresh control | Batch request | Constructor/reuse | Exact raw chunks |
|---|---:|---:|---:|---:|
| GC010 western street | 190.76s | 253.04s | 132.597s, load | 14/14 |
| GC033 shop display | 193.33s | 105.83s | 0.005s, reuse | 14/14 |
| GC040 beach | 186.56s | 108.77s | 0.005s, reuse | 14/14 |

All three requests complete with one constructor. Every native-camera and finite-output audit passes. No model hooks remain after a request, DiT parameter version counters remain unchanged, and the text encoder returns to CPU. Prompt embeddings are recomputed and VAE state cleared between requests. The fresh-reference comparison also checks memory policy and generation settings.

The worker takes467.65seconds for all three requests. The separate fresh controls sum to570.65seconds, but this is not a randomized timing comparison: the first batch load was slower under different I/O conditions. The two warm requests directly demonstrate removal of repeated132-second checkpoint construction. These are throughput results; generated frames are unchanged, and world coherence scores do not improve from this optimization.

This validation covers mixed scenes and moving-camera paths. **All three rendered videos have the same832×464 resolution**, so a different-aspect-ratio reuse claim remains unverified. The first WAIT-only probe and this moving-scene test together validate the current GC research use case.

Evidence: [verification and reference manifest hashes](verification.json), [group result with per-job hashes and cleanup](group-000.json), [complete accounting](summary.json), and the three per-request `run.json` files. The matching fresh videos are available in [GC010](../task-context-validation/GC010/reference18_read4/rollout.mp4), [GC033](../task-context-validation/GC033/reference18_read4/rollout.mp4), and [GC040](../task-context-validation/GC040/reference18_read4/rollout.mp4). All candidate chunks and aggregate videos were committed to the Modal volume as each request completed; the volume directory is recorded in the group result.

Run the concrete [batch launcher](../../docs/persistent-native-batches.md) with the [declared plan](../lingbot-batch-validation-plan.json). It groups exact initialization configurations, runs1–4 serial requests per worker, retains failed jobs and reloads their pipeline before continuing, and keeps multiple groups parallel. Default rollout behavior is unchanged; no persistent HTTP service or new model training is introduced.
