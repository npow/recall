# Two-agent world-model video experiments

## Current demo: actual cross-agent visual memory

The main slides and two-minute video now use **actual shared archived visual KV**.
Both agents start at the same image and camera origin, explore different
directions, then look toward one another's earlier view. Pose-matched canonical
KV is injected into the reader's self-attention at all 40 layers, preserving
the native 18-frame budget. The live caches remain private.

Three cases completed (GC007, GC011, GC026), with real peer injection verified
in both directions. The video radar shows commanded FOV, earlier view directions
and actual 40-layer injection flags. These are serial static-scene visits, not
a simultaneous simulator or a scored consistency improvement.

See the [shared-KV protocol and proof](../reproduction/shared-kv-revisit/README.md).
The previous shared-caption character clips below are retained in the appendix
as earlier prototypes; they are not presented as shared-KV results.

## Selected presentation examples

The earlier deck and two-minute video showed three selected paired scenarios: basket
handoff, greeting, and joint attention. Six paired attempts (12 individual
LingBot rollouts) were reviewed. Three new attempts ran concurrently on three
ephemeral H200 workers. All jobs have stopped.

Selection was by eye for identity stability and readable motion. The later wave
duplicated a character; the high five became clapping or missed contact. Those
attempts were excluded from the presentation. The selected clips still have
cross-view timing and action disagreements and do not establish a shared-world
consistency success rate. The full selection ledger is
[`multi-agent-selection.json`](../reproduction/multi-agent-selection.json).

Presentation media: `docs/slides/media/multi-agent-{pov,wave,point}.mp4` and the
12-second `multi-agent-montage.mp4`. `scripts/build_agent_montage.py` records the
chosen excerpts; `scripts/build_submission_video.py` places them after the
measured RECALL results while keeping the final narration/video at 120 seconds.

## Required presentation: labeled first-person tiles

The deliverable is one MP4 showing **Agent A — Orange POV** on the left and
**Agent B — Purple POV** on the right. Labels remain visible throughout.
The camera is at each character's seated eye position: Orange sees Purple,
and Purple sees Orange. This is not two crops of the earlier observer video.

```bash
.venv-lingbot/bin/modal run --profile nissan-pow scripts/modal_two_agent_pov.py
# Main deliverable: outputs/two-agent-pov/tiled.mp4
# Individual streams: outputs/two-agent-pov/orange.mp4 and purple.mp4
```

This requires the generated first-person starting images already present at
`outputs/two-agent-pov/inputs/orange.png` and `purple.png` in this workspace.
They were made with the built-in image-generation tool using the original
OE025 photo, followed by the reverse viewpoint using both the photo and the
first generated view as references. The final reference specifications were:

- **Orange POV:** seated eye-level view from Orange looking west at Purple;
  own orange floral sleeves and hands in the foreground holding the one empty
  round woven basket with a tall arched handle; Purple opposite in her purple
  kimono, pale green obi, pink hair flower; river on the right.
- **Purple POV:** reverse seated eye-level view looking east at Orange; own
  purple sleeves and empty hands on the knees in the foreground; Orange
  opposite in her orange kimono, silver obi, pale green hair flower, holding
  the same basket in her lap; river on the left. Both images use the same sunny
  riverbank, stone embankment, houses and trees; no text or duplicate basket.

`worldsync/pov_video.py` projects a single versioned symbolic schedule into
view-specific text. Both histories render 12 chunks (189 frames), changing
conditioning at identical chunk boundaries: hold at 0, offer at 2, transfer
at 4, received at 7. Each stream is a real LingBot causal-fast rollout at
16 fps using the same pretrained weights and its own noise seed and KV history.
The H200 worker loads the model once and renders the two complete histories
serially, then downloads both. The final tiler verifies frame count, duration
and cadence before composition and preserves the entire image in each panel.
Each tile uses the same source frame index. The earlier basket pickup is
omitted: both starting views agree that Orange already holds the basket.

**Scope:** shared scripted intent and synchronized presentation, with independent
visual histories. The starting views are generated approximations, not camera
poses from a calibrated 3D scene. There is no shared multiview latent, exchanged
KV archive, closed-loop embodied policy, or past.dev IPC in this render. Visual
agreement and basket ownership must be reviewed in the actual frames. The
symbolic schedule is not a guarantee of physical consistency, and matching
frame indices does not guarantee matching action progress in two generators.

First tiled render (2026-11-15): Modal directory
`two-agent-pov-1794777540538946834`; output `outputs/two-agent-pov/tiled.mp4`,
1664×522, 189 frames, 16 fps, 11.8125 seconds. Both panel labels are visible
throughout the sampled review. Orange's view sees Purple, and Purple's view
sees Orange. The basket remains visible and the characters reach toward it,
but the final intended release is not established: Orange's hands continue
gripping in Orange's view while Purple's view also shows hands around the
basket. Treat the handoff as visually unresolved, not a validated common
ownership transition. Review sheet: `outputs/two-agent-pov/review.jpg`.
Fourteen targeted scheduling, projection and compositor tests passed.

## Earlier shared-camera world-model video

The video path uses the official LingBot-World2 14B causal-fast checkpoint to
generate a continuous MP4 from PlayWorld's OE025 riverbank reference image.
Two character controllers coordinate moving the single basket from the orange
kimono character to the purple kimono character. Their messages and committed
symbolic state produce scheduled text conditioning at chunks 0, 6, 12 and 18.
The same causal generation history persists across all four phases.

```bash
.venv-lingbot/bin/python scripts/prepare_multi_agent_video.py --scripted
.venv-lingbot/bin/modal run --profile nissan-pow scripts/modal_multi_agent_video.py
# MP4: outputs/multi-agent-world-video/rollout.mp4
```

This launches one H200 job with a 20-minute timeout, generates 24 chunks
(23.8125 seconds at 16 fps), and downloads the MP4 and its provenance. It reuses
the existing staged, previously verified checkpoint and checks its byte inventory.
The job is ephemeral and does not deploy a persistent GPU service.

The initial run uses **scripted controllers**, explicit local messages, one
shared camera, and open-loop text conditioning. It uses no past.dev IPC. The
optional LLM planner hit the Anthropic account usage limit before making a plan.
Omitting `--scripted` uses that planner when the account becomes available.
Even with the LLM planner, progress actions are constrained by the scenario;
these are not autonomous embodied policies driven by generated observations.
This render is an experiment in whether LingBot follows the joint actions.
The committed symbolic handoff alone does not establish a visual handoff.
It does not provide two independent first-person views or cross-view validation.

First render (2026-11-15): Modal result directory
`multi-agent-video-1794775853114000446`; 381 frames, 832×464, 23.813 seconds.
The characters visibly handle a basket, but the model invents a second basket
while retaining the original on the ground. Late frames also contain oversized
ducks. The intended single-object handoff is therefore **not visually validated**.
Local artifacts are in `outputs/multi-agent-world-video/`, including the original
`rollout.mp4`, a smaller `demo.mp4`, the full run manifest and a sampled review sheet.
This is actual world-model output and a failed consistency experiment, rather
than evidence that text-conditioned symbolic state controls object permanence.

## Earlier CPU coordination prototype

The runnable first step is a **scripted coordination prototype**, using two real
OS processes and a small top-down world. It does not run LingBot or an LLM policy
and is not evidence of cross-view video consistency. It runs on a CPU.

```bash
python -m worldsync.multi_agent_demo --output outputs/multi-agent
# Open outputs/multi-agent/demo.gif
```

Use a fresh output directory for each run. Pillow is the only dependency beyond
the Python standard library. The demo also writes `trace.json`, `events.jsonl`,
`world.sqlite` and `past-outbox.jsonl` into that directory.

Alice picks up a red parcel, carries it to a meeting point, and puts it down.
Bob still remembers the original location. His action against that old revision
is rejected. He refreshes, walks to the new location, picks up the same parcel,
and hands it back to Alice. Both finish with identical state at revision 10.
The GIF shows the authoritative world alongside both agents' last observations.
Private observations may lag; committed ownership never forks.

## Consistency and IPC

Each agent has a process, a local observation, and its own database connections.
The parent sends scripted steps over multiprocessing queues to make the story
reproducible. Agents read and submit world actions through SQLite. They do not
exchange Python objects directly with each other.

`BEGIN IMMEDIATE` serializes commands across processes. Commands carry the world
revision used to choose the action, and a unique retry token. A stale revision
or invalid action changes no world state. A valid action updates the snapshot
and appends its receipt in one transaction. A repeated token returns the original
receipt; reusing it for different command contents fails. Rejected commands are
also recorded but do not advance the revision. Physical preconditions enforce
one parcel, one holder, adjacent movement, and co-location for handoffs.

Global revision checking intentionally favors simplicity over throughput: even
unrelated changes invalidate a command. For a larger world, move to per-entity
versions with transactional checks on every entity involved in an interaction.
SQLite is suitable for this single-host demo; distributed deployment needs a
world service backed by a transactional database. This prototype has no network
authentication or agent identity security boundary.

## Where past.dev fits

past.dev accepts timestamped text, processes it asynchronously, and recalls ranked
memories with source evidence. Its documented contract does not supply the
conditional world updates used here. Use it for durable shared experience and
semantic discovery; validate plans against live world state before executing.
See the [Memory API overview](https://past.dev/docs/memory-api/overview) and
[API reference](https://past.dev/openapi.json).

The demo reuses `worldsync.research_memory.PastClient`. Its outbox contains one
immutable, timestamped source per committed revision, namespaced by a unique run
ID. Rejected intentions never become facts in memory. By default nothing is sent.
To send these synthetic world events, set `PAST_API_KEY` and run:

```bash
python -m worldsync.multi_agent_demo --output outputs/multi-agent-past --publish-past
# Retry publication from the existing outbox with stable idempotency keys:
python -m worldsync.multi_agent_demo --output outputs/multi-agent-past --publish-past --publish-only
```

Receipts go to `past-receipts.jsonl`; queued receipts do not mean that recall is
ready. Use `PastClient.ingestion(ingestion_id)` to check processing completion.
Use a dedicated project key for isolation from research findings. This demo uses
project-default visibility; a multi-user deployment needs explicit audiences.

After processing completes, both agents can independently recall the same run:

```python
from worldsync.research_memory import PastClient

client = PastClient()
# Copy the run ID from past-outbox.jsonl's metadata.run_id.
for actor in ("alice", "bob"):
    evidence = client.recall(
        "In courtyard run RUN_ID, who moved the red parcel and who holds it now?",
        identity=f"worldsync-{actor}",
    )
    # Treat evidence as context, then read World(...).read() before acting.
```

Do not send frame tensors or KV caches through text memory. Keep visual assets in
an archive/object store and record their references, poses, entity versions and
world revision in the shared event stream. Persisting an event to past.dev is
separate from committing it to the world; a production publisher needs a durable
outbox written in the same transaction as the world event. This demo exports its
outbox after the run and can regenerate evidence from SQLite history if needed.

## Extending RECALL's generated world

The current `lingbot_rollout.py` runs one camera and one generation history.
Neither a shared seed nor exchanging captions establishes shared geometry across
two independent histories. A practical staged demo would proceed as follows:

1. **One authoritative scene, two policies.** Replace each scripted worker with
   an LLM policy consuming its observation and optional past.dev evidence. Keep
   the action validation and trace unchanged. Initially use alternating turns so
   every visible change has an unambiguous committed order.
2. **A shared renderer, alternating viewpoints.** Start both cameras in one
   calibrated coordinate frame, initially near the same pose. Serialize their
   requests on one GPU. Each render gets a committed snapshot, observer pose and
   revision. `render_context()` and the per-frame `contexts` in `trace.json`
   provide that input contract. Add a dedicated experimental renderer adapter;
   do not route synthetic state into the benchmark's restricted director inputs.
3. **A shared visual archive.** Index frames by world ID, pose, revision and the
   versions of visible entities. Retrieve the other agent's compatible views,
   and recompute conditioning for the target camera. `MemoryV2Recompute` is a
   starting point for recomputation, but it currently assumes one trajectory;
   cross-camera alignment and conditioning require new implementation and tests.
   Directly copying another agent's KV cache is not a validated interface.
4. **Invalidate changed evidence.** When Alice moves the parcel, an old frame is
   still evidence of the courtyard but contains an obsolete parcel location.
   The initial safe rule is revision-level invalidation (the prototype emits a
   revision-scoped memory namespace). This loses useful static memory. Reusing
   static regions while excluding changed objects needs masks or a structured
   scene representation, and likely stronger renderer conditioning.
5. **Measure both layers.** Require exactly one accepted pickup in a same-object
   race, no lost updates, idempotent retries and final agent convergence. Separately
   score whether both generated views show the same object identity, committed
   location and ownership. Correct database state cannot guarantee correct pixels.

The visual target is a split screen: Alice moves the parcel while Bob looks away;
Bob turns back and sees it at the meeting point, then they hand it over. Compare
isolated histories, shared text only, and revision-aware shared visual memory.
Use one camera position first to test appearance transfer, then widen the baseline
to test viewpoint consistency. Report rendering latency and stale-frame failures.
Independent simultaneous first-person views with mutual visibility are a later
milestone, rather than something current single-camera RECALL already supports.
