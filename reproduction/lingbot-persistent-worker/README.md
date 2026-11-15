# Reuse pretrained weights across requests

**A repeated ten-chunk memory request falls from210.44 to92.78seconds while preserving every raw-pixel hash.** A warm plain request takes69.04seconds versus165.46seconds for the separate fresh-process control, again matching all ten hashes. This improves experiment throughput by avoiding checkpoint loading; the renderer's mathematical operations are unchanged.

One H100 process ran four serial requests with exactly one constructor. The factory accepts only identical initialization settings and returns the same pretrained pipeline. All use Torch2.6/CUDA12.4 SDPA, stock18/sink6, the original checkpoint and four denoising steps. Each request generates157frames on the ten-WAIT native path from GC007. Prompt context is deliberately recomputed, preserving encoder placement; the probe does not depend on reusing cached prompt embeddings.

| Request, in execution order | Full request | First chunk | Constructor/reuse |
|---|---:|---:|---:|
| Cold four-view memory, seed42 | 210.44s | 128.32s | 104.011s |
| Reused plain model, seed43 | 70.46s | 18.94s | 0.006s |
| Reused four-view memory, seed42 | 92.78s | 18.90s | 0.006s |
| Reused plain model, seed42 | 69.04s | 17.65s | 0.006s |

The cold/reused memory42 pair matches all10 raw-pixel hashes despite the intervening different seed and policy. The final plain42 request matches all10 hashes from the separate fresh Torch2.6/H100 control, after both memory and different-seed requests. Seed43 changes the output. These checks cover state isolation for this configuration and trajectory; they do not establish arbitrary concurrent-request safety.

Every request passes complete native-camera and finite-output checks. No model hooks remain afterward, DiT parameter version counters remain unchanged, and the text encoder returns to CPU. The native generator reinitializes self/cross-attention KV caches and seeded noise; the factory additionally clears prompt and VAE feature caches before reuse. The four-view memory hooks and CPU archives are created and removed per request. No training is performed.

The repeated memory request has2.27× trial throughput versus the cold request in the same worker. The final plain request has2.40× throughput versus the external fresh control. Different-container cold I/O affects the latter comparison, but the directly measured104-second constructor disappearing on reuse explains the substantial saving. All four requests take444.08seconds in this worker. This is not a2.4× improvement in warmed denoising speed and does not improve world coherence by itself.

Evidence: [result and all pixel hashes](result.json), four per-request manifests, and the [frozen plan](plan.json). [Cold memory clip](cold-memory42/rollout.mp4) also represents the pixel-identical reused memory request. The final plain42 pixels match the [existing fresh control](../lingbot-runtime-benchmark/sdpa26/rollout.mp4). The changed-seed output is [plain43](reuse-plain43/rollout.mp4). Every original chunk and full video remains in the Modal directory named in the result.

The secondary workspace rejected H100 registration because a payment method was required; no GPU experiment ran there. [The attempt ledger](attempts.json) preserves that outcome. The successful worker used the established `nissan-pow` workspace and its existing verified weights without another download.

Reproduce with `.venv-modal-new/bin/modal run --profile nissan-pow scripts/modal_lingbot_persistent_worker.py --workspace nissan-pow`. This is an experimental factory wrapper; the standard experiment runners have not yet been converted to persistent services.
