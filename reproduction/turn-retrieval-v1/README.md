# Historical-view retrieval on realized turn returns

The unchanged virtual-position PoseKVMemory prototype improves GC007 seed42 on both short and delayed pure-turn paths. Its earlier translation-path experiment failed; that result remains published. These are development mechanism tests, not PlayWorld scores.

Archive16 early non-sink key/value frames on CPU, select4 by commanded pose overlap, reapply native spatial RoPE and recent virtual temporal positions, and attend to sink6 + retrieved4 + tail8 = the same18-frame budget as stock. The archive is bounded at19.765GB; peak GPU allocation is about85.3GB versus stock80.4GB. This is a prototype inspired by WorldKV/WorldTrace, not a paper reproduction.

On the7.8125-second path, native camera audit and four-chunk exact pixel prefix pass. Endpoint DINO rises0.579→0.738; a blinded full-video review prefers retrieval for preserving the returning roof, sign and facade. Frame PSNR declines and framing still drifts.

On the15.8125-second path with an eight-second away-view hold, native audit and five-chunk exact prefix pass. Both rolling cache18 and cache34 fail to recover the original building. Retrieval returns through the same intermediate landmarks and restores the original Primemall identity continuously, according to the high-confidence blinded review. It then overshoots left: the final frame crops the recovered building to the edge. Endpoint DINO consequently falls0.613→0.516, while PSNR rises7.565→8.244dB. We report this disagreement explicitly: it is a visual memory gain with imperfect camera closure, not improved final-frame reconstruction. Judge evidence and complete videos are retained for inspection.

Warmed generation takes47.59s versus37.58s stock on the short path, and101.56s versus82.46s on the delayed path. CPU archive transfers add overhead. No latency speedup is claimed. The method stores only an early segment and uses intended poses rather than reconstructed actual camera poses; it is not yet a general spatial memory map.

The method is frozen for the new-seed, image-held-out and32-second tests in ../memory-validation-plan.json. Exact native-key positions are also being tested as a separate positional ablation; those results do not change the frozen candidate. No training, image editing, or output frame pasting is used.
