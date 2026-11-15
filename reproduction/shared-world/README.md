# Shared world on past.dev (Phase 0, offline)

Goal: two agents inhabit the same generated world and share what they have seen.

- **Same world.** A world id hashes the source and checkpoint revisions, initial image, caption and seed
  (`worldsync/shared_world.py`, `world_id`). Rollouts with the same id start identically.
- **Frames.** Each chunk's last generated frame is stored as a JPEG blob in a local SQLite database
  (`outputs/shared-world/world.db`, ignored by git), addressed by sha256. Frames are decoded from the video into
  memory; no frame files are written.
- **past.dev.** Text only: one observation per agent and chunk (agent, world time, commanded heading/position,
  optional caption, `frame:sha256:<hex>` reference, optional 160 px base64 thumbnail in metadata), one world
  definition, and one sync fact per agent pair. World time is the data point timestamp, so a recall can be made
  as of an agent's current world time (`queryTimestamp`). Observations can be kept private with an audience.
- **Lookup.** `nearby` is an exact local lookup by heading; `recall` asks past.dev and resolves the frame
  references against the local database.

```
python -m worldsync.shared_world ingest-run reproduction/full-history-GC010 --agent agent-a --dry-run
python -m worldsync.shared_world ingest-run reproduction/commit-forcing/GC010-seed42-h200/stock --agent agent-b --dry-run
python -m worldsync.shared_world compare reproduction/full-history-GC010 reproduction/commit-forcing/GC010-seed42-h200/stock --agents agent-a agent-b --dry-run
python -m worldsync.shared_world nearby w-75c10516056f3799 --heading 90 --exclude agent-b
python -m worldsync.shared_world recall w-75c10516056f3799 --heading 90 --identity agent-b --at 12.75   # needs PAST_API_KEY
```

## Dry run on two existing GC010 rollouts

Agent A is the native58 full-history run, agent B the stock18 run: same world id `w-75c10516056f3799`, same
actions. The compare step records that their raw chunk pixel hashes match through chunk 3 and first differ at
chunk 4, so A's later observations do not hold for B. The dry run queues 30 data points (one world definition, 28
observations with thumbnails, one sync fact), 164,858 bytes, about 486 ingest credits. Nothing was sent; no key was
available.

Caveat found: frames decoded from `rollout.mp4` do not keep the raw pixel parity. For the chunks where the raw hashes
match, the decoded JPEGs are identical for chunks 0 and 3 but differ for 1 and 2, because the whole video is lossy
encoded. Frames for shared memory should be captured from the decoder output during rollout, not from the mp4.

## Not done

- No live past.dev send or recall yet.
- No captions: there is no VLM key here, so observations carry pose and frame references only.
- No generation uses the shared memory yet. Phase 1 would let agent B retrieve agent A's views at matching
  headings during its own rollout.
