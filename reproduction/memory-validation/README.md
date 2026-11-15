# Archived-view memory beyond the rolling cache

The frozen four-view retrieval method improves returning landmark identity in all three predeclared validation pairs. These cover two scenes, a new seed, an unused held-out image, and a 31.8125-second horizon. They are mechanism tests rather than task-matched PlayWorld scores.

| Scene / seed | Duration | Stock endpoint DINO | Memory endpoint DINO | Complete-video finding |
|---|---:|---:|---:|---|
| Mall GC007 /43 |15.8s |0.597 |0.766 | Memory retains the roof, facade and return ordering; exact starting viewpoint remains uncertain |
| Fountain GC006 /42, held out |15.8s |0.885 |0.083 | Memory restores original landmarks, then overshoots toward a bench; stock morphs the fountain and surrounding layout |
| Mall GC007 /42, long delay |31.8s |0.574 |0.746 | Memory recovers original building and route; stock substitutes buildings and exhibits color/sign drift |

Blinded full-video reviews prefer memory in all three pairs (medium, high and high confidence). Native camera audits pass and raw pixels match until the first memory intervention. The generator retains the same 18-frame attention budget: sink6 + retrieved4 + recent/current8. It archives16 early non-sink latent key/value frames on CPU, bounded at19.765GB. No training or output editing is used.

The fountain result is a useful disagreement: the final-frame metric heavily penalizes the changed heading toward the bench, while the full return contains the original fountain, barrel, bench and background arrangement. Stock ends on a fountain and has high semantic similarity, but its detailed structure and surrounding scene change. We report the poor endpoint score rather than substituting a favorable metric. Exact pose closure is not solved.

On the long horizon, warmed generation takes 171.61 s stock versus 210.32 s memory (+22.6%). GPU peak allocation rises80.77→85.71 GB (+6.1%); CPU archive memory is additional. Timing is exploratory, one run per mode. The long hold still contains sign/texture changes, and the scene map covers only early observations. This is not proof of minute-long general worlds, dynamic-process consistency, or a PlayWorld leaderboard improvement.

A32-chunk camera-audit false positive interrupted the first batch. The failure and cancelled attempt are preserved in attempt-ledger.json and failed-long-camera-audit-run.json. The resumed run confirms that the apparent BF16 mismatch came from a 5.96e-8 FP32 difference; native conditioning remained unchanged. Completed generations were reused and only missing calls were retried. See ../camera-audit-v2.md.

Each case directory includes full videos, contact sheets, endpoint PNGs, native manifests and complete blinded assessments. The original frozen plan is ../memory-validation-plan.json. Task-matched released-rubric comparisons are separately declared in ../task-memory-benchmark-plan.json. The held-out fountain image has now been consumed and must not become a tuning case.
