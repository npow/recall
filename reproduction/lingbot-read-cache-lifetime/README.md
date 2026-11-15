# Release temporary retrieval reads before decoding

`PoseKVMemory` used to retain each layer's retrieved GPU keys and values until the next chunk. The VAE decoder never uses these tensors. The rollout now releases them after the clean denoiser pass and before decoding. CPU history and the native persistent KV cache remain in place.

The 32-chunk GC007, seed42, four-nearest-view H100 rerun matches **all 32 original raw-pixel hashes exactly**, including the delayed revisit. Checkpoints, native camera inputs, sampler, seed, hardware and encoder placement match. The protocol name changed from v5 to v6 between runs; fixed controls bypass the player, and the direct pixel comparison verifies this path.

| Measurement | Previous | Release before decoding |
|---|---:|---:|
| Peak allocated GPU memory | 74.347 GB | 72.758 GB |
| Warm 31-chunk suffix | 218.59 s | 227.48 s |
| Temporary GPU reads held during VAE decode | Up to 4.941 GB | 0 |

The overall peak falls by 1.588 GB rather than 4.941 GB because another stage sets the new peak. This is a memory optimization; this single timing comparison shows no speed improvement. Both runs complete, and no weights, attention policy, precision or generated pixels change.

The native-attention integration test also verifies that release preserves persistent KV tensors and CPU archive tensor identities for all retrieval variants.

Evidence: [comparison.json](comparison.json), [run.json](run.json). Original video and its manifest: [nearest4](../multiview-retrieval/nearest4/). The new complete video remains in the Modal results volume under the candidate directory recorded in the comparison.

Reproduce with `modal run --profile nissan-pow scripts/modal_lingbot_read_lifetime.py::main`.
