# Seed noise of the stock arm, measured on the local Thor GPU (in progress)

Status at the time of this commit: the runner is generating, no episode is scored yet. This file is overwritten
by `scripts/thor_seed_noise.py --write reproduction/seed-noise-thor.md` once seed-43 episodes are scored.

What is running (all local, no Modal spend):
- `scripts/thor_select_cases.py`: 12 cases, the first 3 per split in the full run's salted-hash order
  ('worldsync-fullrun-v1'), skipping the 6 insight development cases: GC035, GC022, GC023, IF004, IF010, IF025,
  insight OE025, OE033, OE007, outsight OE028, OE059, OE031.
- `scripts/thor_seed43_runner.py`: stock 18/6, seed 43, the full run's frozen flags
  (`--player-retries 2 --text-prompt both --offload-text-encoder`, 40 chunks; insight `--text-prompt caption`, 60 chunks),
  one episode at a time, cheapest first (IF, GC, out-of-sight, then insight), resumable, pauses if another local GPU job starts.
  About 30-35 min per 40-chunk episode on Thor (about 47 s per chunk), so about 10 hours for all 12.
- `scripts/thor_score.py --watch`: Gemini Vertex judge (gcloud login) on each finished episode.
- `scripts/thor_seed_noise.py`: compares seed 43 with the full run's seed-42 stock scores on the same cases.

Artifacts (git-ignored): `outputs/thor-seed43/episodes/`, `outputs/thor-seed43/gemini/`, logs `runner.log`, `score.log`.
