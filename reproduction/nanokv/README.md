# NanoKV compression-only integration

The [released NanoKV implementation](https://github.com/jiacheng-lu03/NanoKV) includes a LingBot-World2 codec. Its reported14.18× effective cache compression preserves fidelity to a matched stock rollout; it does not establish a PlayWorld improvement or longer-history quality. Reported end-to-end LingBot latency is slightly worse than stock. We start with compression only, without Fast readers.

The527MB codec was downloaded from immutable HF revision `4b0dac1219ed72ead539f8d063348949e6aeb045` and verified against published SHA256 `b014909923f657b8019203bd581da77e8c6e0880bf7aa80b3eadfbf6a433f458`. A weights-only CPU load verified80 K/V entries,40layers, one sequence-parallel rank, rank640 and5120-wide means. See [artifact verification](artifact-verification.json). Weights stay outside Git.

The published profile retains18/6 logical cache slots; compression alone does not extend history. Our planned experiment uses58 slots to retain the entire positive GC010 native58 control. First require an identity codec to reproduce every native58 raw video chunk, then measure compressed18 fidelity against stock18 and compressed58 against native58. The generator, sampler, camera adapter and VAE remain the original implementation; only instance-local cache/attention bindings change during the isolated diagnostic. No reader or existing retrieval method is combined initially.

Status: artifact verified; adapter implementation and CPU tests underway. No NanoKV GPU parity, storage, latency, return or benchmark improvement is established yet. The GPU proof launcher must preserve failed attempts and gate compressed runs on successful identity parity. Calibration on the released short18-slot setup does not guarantee quality at58 slots.
