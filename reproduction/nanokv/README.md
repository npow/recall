# NanoKV compression-only integration

The [released NanoKV implementation](https://github.com/jiacheng-lu03/NanoKV) includes a LingBot-World2 codec. Its reported14.18× effective cache compression preserves fidelity to a matched stock rollout; it does not establish a PlayWorld improvement or longer-history quality. Reported end-to-end LingBot latency is slightly worse than stock. We start with compression only, without Fast readers.

The527MB codec was downloaded from immutable HF revision `4b0dac1219ed72ead539f8d063348949e6aeb045` and verified against published SHA256 `b014909923f657b8019203bd581da77e8c6e0880bf7aa80b3eadfbf6a433f458`. A weights-only CPU load verified80 K/V entries,40layers, one sequence-parallel rank, rank640 and5120-wide means. See [artifact verification](artifact-verification.json). Weights stay outside Git.

The published profile retains18/6 logical cache slots; compression alone does not extend history. Our planned experiment uses58 slots to retain the entire positive GC010 native58 control. First require an identity codec to reproduce every native58 raw video chunk, then measure compressed18 fidelity against stock18 and compressed58 against native58. The generator, sampler, camera adapter and VAE remain the original implementation; only instance-local cache/attention bindings change during the isolated diagnostic. No reader or existing retrieval method is combined initially.

Status: artifact and adapter CPU checks passed. H200 identity58 now reproduces **all14 native58 raw chunks exactly**, with matching settings, unchanged backbone parameter versions and restored temporary bindings. See [GPU proof](identity58/result.json) and [frozen identity plan](identity58-plan-v1.json). This validates the lossless adapter path, not the lossy codec. Compressed18 and compressed58 runs are submitted under [a frozen plan](compression-plan-v1.json); their GPU codec smoke precedes full model loading. No lossy-codec storage, latency, return or benchmark improvement is established yet. The GPU proof launcher must preserve failed attempts and gate compressed runs on successful identity parity. Calibration on the released short18-slot setup does not guarantee quality at58 slots.

## Completed matched14B diagnostic

| Arm | Cache payload + operators | Peak allocated GPU | Cold elapsed | Gemini GC |
|---|---:|---:|---:|---:|
| Native58 |69.18GB occupied raw history|121.39GB|205.2s|5.0|
| Compressed18 |1.57GB|56.99GB|200.1s|1.0|
| Compressed58 |3.44GB|57.33GB|223.1s|2.8|

All14identity58 chunks match native58 exactly. Compressed18/58 also match each other through chunk3, diverging at the first18-slot eviction on chunk4. Both use unchanged backbone weights and the same codec. Compressed58 recovers recognizable original street buildings; compressed18 ends on an invented blue saloon. Framing, lettering and object identity remain imperfect. Gemini reports a return but penalizes identity/texture in compressed58; memory savings do not imply equal fidelity. [Matched summary](matched-summary.json), [frame comparison](matched-contact.jpg), and [full videos](codec58/rollout.mp4).

The cache+operator reduction is20.09× relative to occupied uncompressed history; total peak GPU allocation falls52.8%. These are measured on this single14-chunk case. Runtime is one cold run each, not a replicated speedup measurement. Stock18's separate proof run retained extra validation snapshots, so its peak is not used for a clean memory comparison.

A frozen32-chunk mall stress comparison tests logical58 versus130 on seeds42and43 together;130can retain all128latentframes. Its initial launcher failed before preflight because of a Modal dependency-count mismatch; that log is preserved. The corrected launch uses the same frozen model plan. Longer-history quality and adaptive PlayWorld improvement remain unproved.
