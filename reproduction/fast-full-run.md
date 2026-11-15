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

## Measured throughput

(see below; filled in from the 12-episode measurement run)
