# Fast full run: warm adaptive-episode queue + scoring pipeline

Goal: 171 cases x 2 arms (about 342 adaptive, player-driven episodes) inside the submission window
(due 2026-11-15 2 pm PST) without paying the 112 s model load and a container start per episode.

## What changed versus the old cohort runs

| Old (`modal_lingbot_powered.py` style) | New (`modal_lingbot_adaptive_runner.py`) |
| --- | --- |
| one container per episode (cold start + 100-155 s model load every time) | warm container loads the 14B pipeline once, then drains a `modal.Queue` |
| a separate fan-out per seed/arm | one queue per arm, N drain workers per arm, episodes autoscale across them |
| a failed player call (`PlayerResponseError`) lost the episode | transient failures are retried in the same warm container (cap 3), each attempt kept as `__a<k>` |
| no resume | `episode_done()` checks the volume first; re-running `submit` skips finished episodes |
| download-all, then score | scoring is a pipeline stage: each finished episode is fetched (only `rollout.mp4` + `run.json`) and judged by Gemini as it lands |
| Opus judge in the generation container | no judge in the container (Gemini only; Opus stays optional) |

Files: `scripts/modal_lingbot_adaptive_runner.py` (Modal app), `scripts/adaptive_runner_client.py`
(submit/status), `scripts/score_pipeline.py` (Gemini stage), `worldsync/lingbot_episode_queue.py`
(retry/resume/throughput logic, unit-tested), `worldsync/lingbot_runner_core.py` (`result_name` for stable
result directories). Arms: `stock` (18/6, H100-style memory settings: expandable segments + text-encoder
CPU offload, no quantization) and `budget` (`--budget-memory` K10 native, H200). Add a frozen config by
editing `ARMS` in `worldsync/lingbot_episode_queue.py` (extra rollout flags per arm).

## Usage

```bash
export MODAL_PROFILE=nissan-pow        # the other workspaces cannot launch GPUs
# 1. deploy once (max containers per GPU class; shared workspace, check what else is running)
ADAPTIVE_MAX_CONTAINERS=5 modal deploy scripts/modal_lingbot_adaptive_runner.py
# 2. queue the work and start drain workers (use the modal tool's python for the client)
MODAL_PY=~/.local/share/uv/tools/modal/bin/python
$MODAL_PY scripts/adaptive_runner_client.py submit --label full1 --arms stock,budget --all \
    --seeds 42 --containers stock=5,budget=5 --canary
# 3. score as episodes land (project venv; gcloud login must be active)
.venv/bin/python scripts/score_pipeline.py --label full1 --workers 12 --watch
# monitor / resume
$MODAL_PY scripts/adaptive_runner_client.py status --label full1
$MODAL_PY scripts/adaptive_runner_client.py submit --label full1 --arms stock,budget --all --seeds 42 --containers stock=5,budget=5   # skips done
.venv/bin/python scripts/score_pipeline.py --label full1 --summarize
```

`--per-split N` picks N cases per split by a salted hash of the case id (never by outcome);
`--cases gc:GC039,outsight:OE021` selects explicitly. The aggregation printed by `--summarize` is the
paper's formula: overall = equal mean of the four dimension means (GC, IF, insight, out-of-sight).

Safety notes: results are written under `adaptive/<label>/<arm>/<split>-<TASK>-seed<N>__a<k>/` plus
`progress/` and `failed/` records; nothing is overwritten. A CUDA fault aborts that drain worker
(remaining episodes stay queued for the other workers; respawn with `submit`). The canary flag runs a
fixed 8-chunk path before and after the queue and compares chunk pixel hashes to catch state leakage.

## Measured throughput (12 episodes, `nissan-pow`, 2026-11-14 22:05-22:35)

Setup: 6 cases chosen by a salted hash of the case id (never by outcome; 2 GC, 1 IF, 1 insight, 2 out-of-sight:
GC039, GC001, IF013, OE022, OE021, OE034), seed 42, both arms, 3 warm containers per arm (6 total), canary on.
Episodes are adaptive and player-driven (Anthropic player calls inside the container); no judge in the container.

| | stock 18/6 ("H100-style" settings) | budget memory K10 native (H200) |
| --- | --- | --- |
| episodes completed / failed | 6 / 0 | 6 / 0 |
| mean seconds per episode in a warm container | **123** (58-251) | **192** (52-about 600) |
| per-chunk wall time incl. player call (log lines) | about 9 s | about 12-16 s |
| episodes per hour per container | 29 | 19 |
| peak GPU memory (canary, fixed path) | 68.7 GB | 72.0 GB (canary has no memory method on) |
| container start to first useful episode | about 3 min once (provision about 110 s, load + warm-up + canary about 70 s) | same, but the first H200 container waited about 10 minutes for a slot |

Caveats: n=6 per arm, one seed. Episode length is set by the player (it ENDs early on short reference plans), so
episode seconds vary 4x between cases; the two arms did not run the same player trajectories, and the budget
mean is inflated by one long insight-type episode. Treat the means as +-40%.

**State-leakage check.** The canary (fixed 8-chunk path, GC007 `turn_return`, seed 42) ran before and after each
container's queue of adaptive episodes: all 6 containers reproduced identical per-chunk pixel hashes before vs after
(3 stock-class containers with hash set `19d26ac1...`, 3 budget-class containers with `39f5dd20...`; the two classes
differ from each other because they ran on different GPU models). No leakage seen.

**Scoring stage.** The 6 stock episodes were downloaded (19 MB each, 114 MB total) and judged by the Gemini
Vertex pipeline with 6 workers in about 100 s end to end (about 2 calls per episode, none failed). Download plus
scoring keeps pace with generation, so scoring is not the bottleneck at 12 workers.
Side observation: IF013 (an interaction task) gated `fail` and scored 1.0, i.e. our camera-only controls do not
attempt IF tasks; the four-dimension fork owns that issue.

## Projection for the full 171 x 2 run (from the numbers above)

Container-hours: stock 171 x 123 s = 5.8 h; budget 171 x 192 s = 9.1 h (use 150-200 s for budget given the noise);
plus about 3 min start per container. Arms run concurrently on separate queues.

| containers (stock + budget) | stock arm | budget arm | wall time for all 342 |
| --- | --- | --- | --- |
| 5 + 5 | 1.2 h | 1.8 h | **about 2 h** |
| 3 + 3 | 1.9 h | 3.0 h | about 3 h |
| 8 + 8 | 0.7 h | 1.1 h | about 1.2 h (if the workspace grants 16 slots) |

For comparison the old one-container-per-episode path measured about 63 episodes/hour on about 10 containers,
i.e. 5.4 h for 342. The warm queue needs about 2 h on 10 containers: roughly **2.7x faster**, with retries,
resume and live scoring as extras. Uncertainty is +-40% (small sample).

GPU cost (assuming Modal list prices of about $3.95/h for H100 and $4.54/h for H200, not verified here):
stock 5.8 h x $3.95 = $23, budget 9.1 h x $4.54 = $41, start-up overhead about $3: **about $65-70**.

Capacity is the real risk, not throughput: the workspace is shared with other forks. In this measurement the
H200 containers waited 7-17 minutes for slots while 8 other H200 containers were running. Plan the full run for a
window where the other forks are idle, and prefer the stock arm on H100-class slots (they were available
immediately). Idle containers hold a slot for `ADAPTIVE_SCALEDOWN` seconds (default 45 s) after the queue empties.

## Known limits

- The arms are `stock` and `budget`; the powered-test extras (`--end-guard`, `--player-retries`, the
  adaptive-ablation flag) are on branch `worktree-agent-a5f22f4098cb528aa`, not on `main`, so they are not in
  `ARMS` yet. Add them to the arm's `extra` list once the config is frozen (the argv is passed straight to
  `worldsync.lingbot_rollout`).
- IF episodes are launched like any other, but camera-only control cannot attempt interaction objectives (see above).
- A Modal H100 request may be served by an H200; hashes are comparable only within one GPU model.
- A CUDA fault aborts that worker (the remaining queue stays for the others). `submit` again to respawn workers.
- `status` uses the modal tool's python (no numpy); `score_pipeline.py` needs the project venv.
