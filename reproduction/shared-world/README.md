# Shared world on past.dev (Phase 0, offline)

Goal: two agents inhabit the same generated world and share what they have seen.

- **Same world.** A world id hashes the source and checkpoint revisions, initial image, caption, seed, frame rate
  and chunk size (`worldsync/shared_frames.py`, `world_id`). The memory method is not part of it, so stock and
  retrieval rollouts from the same start share a world until their pixels differ.
- **Frames.** Each chunk's last generated frame is stored as a JPEG blob in a local SQLite database
  (`outputs/shared-world/world.db`, ignored by git), addressed by sha256. Frames are decoded from the video into
  memory; no frame files are written.
- **past.dev.** Text only: one observation per agent and chunk (agent, world time, commanded heading/position,
  caption when one is given, `frame:sha256:<hex>` reference, optional 160 px base64 thumbnail in metadata), one world
  definition, and one sync fact per agent pair. World time is the data point timestamp, so a recall can be made
  as of an agent's current world time (`queryTimestamp`). Observations can be kept private with an audience.
- **Lookup.** `nearby` is an exact local lookup by heading; `recall` asks past.dev and resolves the frame
  references against the local database.

```
python -m worldsync.shared_frames ingest-run reproduction/full-history-GC010 --agent agent-a --dry-run
python -m worldsync.shared_frames ingest-run reproduction/commit-forcing/GC010-seed42-h200/stock --agent agent-b --dry-run
python -m worldsync.shared_frames compare reproduction/full-history-GC010 reproduction/commit-forcing/GC010-seed42-h200/stock --agents agent-a agent-b --dry-run
python -m worldsync.shared_frames nearby w-8a495fb5ac51a5b8 --heading 90 --exclude agent-b
python -m worldsync.shared_frames recall w-8a495fb5ac51a5b8 --heading 90 --identity agent-b --at 12.75   # needs PAST_API_KEY
```

## Dry run on two existing GC010 rollouts

Agent A is the native58 full-history run, agent B the stock18 run: same world id `w-8a495fb5ac51a5b8`, same
actions. The compare step records that their raw chunk pixel hashes match through chunk 3 and first differ at
chunk 4, so A's later observations do not hold for B. The run queues 30 data points (one world definition, 28
observations with thumbnails, one sync fact), 7,711 bytes of content, 31 ingest credits. Billing counts content
only: the thumbnails in metadata were not charged.

Caveat found: frames decoded from `rollout.mp4` do not keep the raw pixel parity. For the chunks where the raw hashes
match, the decoded JPEGs are identical for chunks 0 and 3 but differ for 1 and 2, because the whole video is lossy
encoded. Frames for shared memory should be captured from the decoder output during rollout, not from the mp4.

## Live run

The 30 data points were sent to past.dev, one ingestion each: 31 ingest credits, matching the estimate. Each became
searchable within seconds; past.dev's derived memories took several minutes more. As agent-b, `recall --heading 90`
returned eight observations, all at heading 90 and from both agents, and every frame reference resolved to a blob in
the local database. With `--at 4.0` only data points at or before world time 4.0 s were returned. A question about
where the two agents stop sharing the world returned the sync fact first.

Recall is semantic, not exact. Use `nearby` for exact heading lookup and past.dev for dated, cross-agent facts;
captions should make past.dev results more useful. `recall_view` drops results that belong to another world.

## Three agents in one LingBot world (demo, 2026-11-15)

`scripts/modal_lingbot_trio.py` runs a viewer and two duelers from the same seed 42 / GC010 start with
RECALL pose retrieval; the viewer exports its K/V archive (`--archive-export`) and each dueler also retrieves from it
(`--foreign-archive viewer=...`, `lingbot_retrieval.select_frames(..., foreign=...)`). World id `w-ea4cc9b26e3af54b`
(the identity now includes `world_text_sha256` when the conditioning text was changed). Ingest + demo:

```
for a in viewer duel-a duel-b; do
  python -m worldsync.shared_frames --db outputs/shared-world/trio.db ingest-run outputs/shared-world/trio/$a --agent $a \
      --captions outputs/shared-world/trio/$a/captions.json --dry-run; done
python -m worldsync.shared_frames --db outputs/shared-world/trio.db compare outputs/shared-world/trio/viewer outputs/shared-world/trio/duel-b --agents viewer duel-b --confirm-send
python reproduction/shared-world/make_trio_demo.py outputs/shared-world/trio   # side-by-side mp4 with memory captions
```

Findings: duel-b's chunks 0-3 are pixel-identical to the viewer's (same seed, same WAIT/RIGHT x3), and at chunk 4,
facing 180 degrees, it attended to the viewer's views 13-16. duel-a never attended to the viewer: retrieval needs
four views within 8 units and S moves 4 units per chunk, and its chunk 0 already differed because its clip was 7
chunks instead of 8 (the seeded noise depends on clip length). 26 data points (34 credits) went to past.dev; recall
as duel-b at heading 180 returned the viewer's and its own sightings with resolvable frame references.
`opponent_cue`/`cowboy_prompt` turn the stored positions into a text placement of the other agent.

## Two agents circling each other (demo, 2026-11-15)

`CameraPlan` gained `CIRCLE_LEFT`/`CIRCLE_RIGHT` (orbit the point 4 units ahead at 45 deg/s, always facing it).
`PATHS["circle_a"]` is one full lap; `PATHS["circle_b"]` walks to the far side of the circle, turns to face its centre,
then orbits half a lap through agent-a's earlier poses. Launcher: `scripts/modal_lingbot_orbit.py` with the cleaned
start image `gc010_clean2.jpg` (minimap/HUD painted out); agent-a exports its archive (`--archive-until 40`), agent-b
retrieves from it (`--foreign-archive agent-a=... --min-retrieved-frames 1`). World `w-55018cb255709d31`.

Findings: chunk 0 is pixel-identical across the two runs. On the far side (headings 180-270) agent-b attended to
agent-a's views at every chunk and generated the same blue SHERIFF building, porch posts and poles from the same
positions. Returning to the start (315, 0) the views keep the style but not the detail, and agent-a's own lap does
not return to the original DRUGS street. past.dev recall as agent-b at heading 225 returns agent-a's sighting from
the identical pose first; 22 data points, 34 credits.

```
python -m worldsync.shared_frames --db outputs/shared-world/orbit.db ingest-run outputs/shared-world/orbit/agent-a --agent agent-a --captions ... --dry-run
python reproduction/shared-world/make_trio_demo.py outputs/shared-world/orbit agent-a agent-b
```

## Coliseum: second agent started from the first agent's halfway frame (demo, 2026-11-15)

`scripts/modal_lingbot_orbit.py --image arena_start_16x9.jpg --image-b arena_b_start_from_a_halfway.jpg`: agent-a
orbits one lap from a generated coliseum image; agent-b starts from agent-a's own generated frame at the far side
(last frame of chunk 4, pose (0, 0, 8) heading 180), with `--start-pose 0 8 180` and agent-a's archive. A first
attempt that gave both agents the same start image and only relabelled agent-b's pose was rejected: its first four
chunks were pixel-identical to agent-a's.

Because agent-b's start image differs, its own identity hash differs; `ingest-run --join WORLD_ID agent-a 4` files it
under agent-a's world `w-c79b5bfa7e3c025d` and records a join fact with the source frame reference. At matched poses
the backdrop agrees (wall, red banners, dark gateways at 180-225, the columned gate at 270-0); the figures do not:
the model draws two or three gladiators in each view. 22 data points, 40 credits.

## Not done

- Captions are hand-written per chunk (no VLM key here).
- Retrieval requires a full set of four views and an 8-unit range, so moving agents rarely use another
  agent's archive; partial sets and slower motion are the proposed next step.
