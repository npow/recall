# Cross-agent visual KV revisit

This is actual RECALL-style peer visual memory, not the earlier shared-caption
prototype. Three GPU diagnostics completed: GC007, GC011 and GC026. Every case
verified peer KV injection at all 40 DiT layers in both directions, on return
chunks 4 and 5. These are functional injection checks, not consistency scores.

Each case starts both cameras at the same image and origin. B scouts right;
A explores left and then looks where B looked; B then looks where A looked.
Visits run serially on one warm H200 per case. The two additional cases ran
concurrently on separate workers. There is no claim of a simultaneous simulator.

| Visit | Camera actions | Archive interaction |
|---|---|---|
| B scout | WAIT, RIGHT, WAIT | publishes the right-hand view |
| A visit | WAIT, LEFT, WAIT, RIGHT, RIGHT, WAIT | reads B's +60° view |
| B visit | WAIT, RIGHT, WAIT, LEFT, LEFT, WAIT | reads A's −60° view |

`SharedPoseKVMemory` captures normalized canonical keys before RoPE and current
values on the native clean-cache pass. A chunk is published only after every
layer was captured. The common CPU archive is bounded to eight frames per actor.
Frames carry actor, source pass, latent index, commanded pose, scene revision
and tensor geometry. A reader excludes its own entries, other revisions and
incompatible shapes, and selects up to four peer frames within 20° and 0.01
translation units of its target pose. These tests use rotation only.

Retrieved keys receive the reader's virtual temporal RoPE positions. The
self-attention window is first 6 + K peer frames + recent (12−K), retaining the
native 18-frame budget and every current query frame. All model weights remain
frozen. Live caches are private; archived visual KV is shared.

`proof.json` and the GC011/GC026 subdirectories contain actual run manifests and
the peer-frame IDs, donors, pose gaps, injection chunks and layer counters.
Video provenance and pixel hashes are in the visit manifests. The radar uses
recorded commanded headings, native-intrinsics horizontal FOV and actual
attention trace flags. It shows aligned replay of serial visits; it does not
estimate realized camera motion from generated pixels.

```bash
.venv-lingbot/bin/modal run --profile nissan-pow scripts/modal_shared_kv_revisit.py
.venv-lingbot/bin/modal run --profile nissan-pow scripts/modal_shared_kv_revisit.py --more
.venv-lingbot/bin/python scripts/render_shared_kv_radar.py
```

The first result is at `outputs/shared-kv-revisit/tiled.mp4`; the radar version is
`docs/slides/media/shared-kv-radar.mp4`. The other cases use
`outputs/shared-kv-revisit-GC011` and `outputs/shared-kv-revisit-GC026`.
The paired traces verify that peer tensors enter attention. Visual agreement
has been inspected qualitatively; no independent-cache control or improvement
score has been run for this prototype.
