# FlashAttention runtime investigation

The existing Torch2.6 SDPA path already dispatches to fused FlashAttention. Its measured self-attention operator accounted for 37.6% of warmed GPU operator time; this investigation upgrades that fused kernel rather than introducing fused attention for the first time.

On one H100, sequential subprocesses compared the existing Torch2.6/CUDA12.4 SDPA bridge, a Torch2.9/CUDA12.8 SDPA control, and FlashAttention4 on the latter stack. Each shape used seven timing blocks, alternating backend order within the newer stack, with 20 calls per block. Cold compilation was excluded from the medians. All tensors were BF16, batch1, 40 heads, head dimension128, with6032 queries.

| Attention keys | Torch2.6 SDPA | Torch2.9 SDPA | Torch2.9 FA4 | FA4 speedup over existing bridge |
|---|---:|---:|---:|---:|
| 4 latent frames | 2.252 ms | 2.387 ms | 1.078 ms | 2.09× |
| 18 latent frames | 9.661 ms | 9.979 ms | 5.136 ms | 1.88× |
| 22 latent frames | 12.473 ms | 12.252 ms | 6.247 ms | 2.00× |
| 34 latent frames | 18.872 ms | 18.797 ms | 10.123 ms | 1.86× |
| 512 text tokens | 0.282 ms | 0.311 ms | 0.167 ms | 1.69× |

FA4 versus same-stack SDPA relative output error ranged from0.00067 to0.00144. Both implementations had about0.0023 relative error against an independent FP32 reference on an eight-query/two-head slice with the complete key sequence. These are small BF16 differences, **not bit-identical outputs**. No generated-video quality or full-model speed claim follows from this microbenchmark.

The measured call includes each bridge's allocation/format handling, so it is not a kernel-instruction-only measurement. In particular, the current SDPA bridge allocates a padded output and copies the attention result into it. FA4's measured temporary allocation was about62MB versus125MB for that bridge. Neither removes persistent KV storage or the model's weights.

FA4's current dependencies failed on Torch2.6: the optional DLPack binary used an incompatible C++ ABI; disabling that optional addon then exposed a newer Torch dtype requirement in Quack. Both failures are retained. The successful benchmark keeps Torch2.6 intact and installs Torch2.9 in an isolated virtual environment. The optional DLPack addon also emitted a fallback warning on the newer environment; the raw report's addon flag indicates the requested package path, not successful fast-addon activation.

Sources: [official FlashAttention implementation](https://github.com/Dao-AILab/flash-attention), [optional TVM-FFI DLPack support](https://github.com/apache/tvm-ffi/blob/main/python/tvm_ffi/_optional_torch_c_dlpack.py). Measurements: [completed benchmark](lingbot-attention-kernel-benchmark.json), [ABI failure](lingbot-attention-kernel-abi-failure.json), [dtype failure](lingbot-attention-kernel-dtype-failure.json). Run `modal run --profile nissan-pow scripts/modal_lingbot_attention_benchmark.py` to repeat.

Follow-up complete: [full-model pilot](lingbot-runtime-benchmark/README.md) measures1.21× warm throughput versus same-stack SDPA and1.16× versus the established stack, including decoding and transport. Cold loading remains substantial. Scored experiments retain the established Torch2.6 runtime; these speed results do not establish long-horizon quality equivalence.
