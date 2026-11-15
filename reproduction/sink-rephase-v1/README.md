# Rejected initial-sink temporal rephasing pilot

GC007, seed 42, native turn-and-return path, stock 18-frame attention budget. Store six canonical pre-RoPE initial keys on CPU. When the commanded camera returns within 2 movement units and 45 degrees of the initial view, rephase those keys just before the rolling tail. Values and persistent native cache remain unchanged. This is an ablation inspired by virtual-position memory work, not a paper reproduction.

Native camera audits pass; all six pre-intervention chunks have identical raw pixels to stock. Intervention fires only in chunks 6 and 7. Unit tests exercise actual native attention and verify that persistent key/value caches are unchanged. CPU key storage is bounded at 3.706 GB, and peak allocated GPU memory is 83.91 GB versus stock 80.20 GB.

**Rejected:** endpoint DINO barely changes (0.5788 to 0.5856), and blinded full-video review prefers neither. Both return to altered signs and an invented roof/building. No credible quality gain. The full-history pilot remains substantially better. A separately predeclared delayed-return stress test also evaluates this hypothesis; those results must be reported even if negative.

Stock artifacts are in ../context34-v1/. Complete candidate video, run manifest and visual assessment are preserved here.
