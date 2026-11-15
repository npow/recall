# Delayed-return stress test: all three variants fail

GC007, seed 42; fixed native camera path: initial hold, three seconds turning right, eight seconds holding the away view, three seconds turning left, final hold. Total video duration is 15.8125 seconds / 64 latent frames. Both cache18 and cache34 evict outbound views before the return. The experiment was declared in ../delayed-return-plan.json before generation.

Stock18, context34 and pose-gated sink rephasing all pass the native camera tensor audit. Context34 matches stock raw pixels until the cache intervention at chunk4; sink rephasing matches stock through chunk13 and intervenes in chunks14–15 only. The long WAIT is intentional, so static frames during it are not themselves a motion failure.

**No successful return:** both larger context and key rephasing pass the starting facade only briefly, drift into altered buildings and finish on another walkway view. The return also contains abrupt blur/re-rendering. The blinded reviewer weakly prefers context34's scene vocabulary to stock's novel blue facade, but explicitly gives neither credit for recovering the starting place. Sink rephasing and stock are a joint failure. This test does not show coherent 16-second persistence.

This bounds the earlier positive result: cache34 helps when the observed outbound history fits, but does not solve memory beyond the rolling window. The next separately predeclared test applies the unchanged bounded historical-view retrieval prototype to these realized pan paths; its earlier translation-path result was negative.

Complete videos, contact sheets, endpoint images, native run manifests and blinded reports for both comparisons are preserved here. Metrics and timing are in the comparison JSON files. These are diagnostic fixed paths, not adaptive PlayWorld scores.
