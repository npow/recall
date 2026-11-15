# Four versus eight views, and view diversity

The user-requested factorial ablation holds attended tokens at 18 latent frames: sink 6 plus historical and recent/current frames. Four historical views leave 8 recent/current frames. Up to eight historical views leave only 4–7 recent/current frames when activated. All four full 31.8125-second GC007 seed42 videos use H100, the same weights, seed, native camera poses and encoder offload. The CPU archive remains the same 16 frames.

| Variant | Endpoint DINO | Blinded full-video result against four nearest |
|---|---:|---|
| Four nearest, reference | 0.745 | Reference: continuous return, landmark retained, heading offset remains |
| Eight nearest | 0.772 | Worse: cuts/resets during the hold and corrupted return geometry |
| Four diverse | 0.807 | Tie: similar continuous return and pose offset |
| Up to eight diverse | 0.766 | Worse: smeared jump at the start of the return and fused buildings |

**Keep four nearest as the validated candidate.** All alternatives have better endpoint DINO but do not improve complete-video coherence. This is why selecting on a final-frame metric alone would be misleading. Four diverse views are promising for finer investigation, but the blinded comparison is a tie and no held-out improvement has been established. Neither eight-view configuration is recommended from this trial.

Diversity means at least 15 degrees or 2 movement units between selected commanded poses. It does not pad the read with duplicates. The eight-diverse mode therefore reads five views during much of the hold and up to eight during the return; actual counts and retained recency are recorded per chunk. The current archive contains only the early scene, and intended poses are not measured visual poses.

These results show a tradeoff at fixed attention capacity: more historical views replace recent temporal context. They do not prove that eight views would be worse with a larger total budget or compressed historical tokens while preserving all recent context. The next useful comparison should preserve that recent window rather than simply increasing historical allocation.

The first eight-diverse run failed during VAE decoding because of CUDA allocator fragmentation. It is retained in failed-eight-diverse-run.json and the attempt ledger. Only this run was retried with expandable allocator segments; all 28 previously completed chunks have exactly identical raw-pixel hashes. The retry finishes all 32 chunks and passes the same camera/prefix audits. This runtime fix changes allocation, not the memory algorithm.

Complete videos, contacts, manifests, endpoint images, metrics and anonymous visual judgments are retained in nearest4/, nearest8/, diverse4/ and diverse8/. These are development ablations, not PlayWorld scores or a held-out selection result. The frozen plan is ../multiview-retrieval-plan.json.
