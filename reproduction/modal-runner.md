# Persistent multi-run Modal runner

Every one-shot experiment used to launch its own container and reload the 14B LingBot-World2
pipeline (about 100-155 s), then generate for under a minute. Load dominated, and each job
held a GPU-concurrency slot. This runner loads the pipeline once per container and serves many
rollouts from it. Nothing about generation changes: same `worldsync.lingbot_rollout` code, same
checkpoint, no quantization.

## Pieces

| File | Role |
| --- | --- |
| `worldsync/lingbot_rollout.py` | `main(argv, runtime)`: the one-shot CLI, now callable in-process. With `runtime=None` it is unchanged. |
| `worldsync/lingbot_runtime.py` | `WarmRuntime`: builds the pipeline once, re-points the rolling-window geometry per job, detaches LoRA after each job, `fingerprint()` weight check. |
| `worldsync/lingbot_runner_core.py` | `RunnerCore.run(job)` (warm) and `run_oneshot(job)` (fresh process, for reference), plus `job_to_argv`. |
| `scripts/modal_lingbot_runner.py` | Modal app: `RunnerH200`, `RunnerH100` (warm classes) and `oneshot_h200`, `oneshot_h100`. |
| `scripts/lingbot_runner_client.py` | Client for the deployed app; threads = containers used. |

## Per-job state reset

Upstream `generate()` already builds fresh KV caches, the cross-attention cache and the noise
(from a per-call seeded generator) on every call, and resets `_cross_attn_initialized`. Our
`main()` creates the hooks, memory bank, camera plan and VAE streaming cache per job and removes
them in `finally`. What a warm process additionally has to manage:

* **Cache geometry.** `local_attn_size` and `sink_size` are int attributes on the pipeline and
  every self-attention module, so a job with another window (e.g. 46/6) re-points them
  (`set_cache_geometry`). Weights are unaffected.
* **Process-global monkeypatches.** The diagnostic translation-scale patch of
  `compute_relative_poses` compounded on a warm process; `main()` now restores it. Memory classes
  restore `model_fast.attention` and remove their hooks in `close()`.
* **LoRA.** `--lora` wraps the q/k/v/o linears; `between_jobs()` puts the originals back.
* **Text encoder.** If a job offloaded it to CPU, upstream moves it back to the GPU when a new
  prompt needs encoding. The T5 prompt cache is kept (same prompt gives the same tensors).
* **Allocator.** `gc.collect()` and `torch.cuda.empty_cache()` between jobs; peak-memory counters
  are reset at job start, so `peak_allocated_bytes` is per job (including the resident weights).
* **Weight integrity.** A float64 sum over four weight tensors is compared after every job
  (`weights_unchanged` in the result).

## Validation: warm results are bit-identical to one-shot

Same job, same code commit, same GPU model (H200), GC007 `turn_return`, seed 42, 8 chunks.
Reference = a fresh process per job (full reload). Warm = one container, jobs in this order:
stock, 46/6, budget memory (K10 native), stock again, 46/6 again, memory LoRA, stock again.
`pixel_sha256` of every chunk compared (`scripts/compare_runner_results.py`;
raw: `reproduction/runner-validation/summary.json`, `compare-h200.txt`):

| Job | One-shot (s) | Warm (s) | Chunk hashes |
| --- | --- | --- | --- |
| stock 18/6 | 189 | 44, 44, 44 (jobs 1, 4, 7) | identical, 4 of 4 runs |
| context 46/6 | 171 | 48, 48 (jobs 2, 5) | identical, 3 of 3 runs |
| budget memory K10 native | 202 | 69 (job 3) | identical |
| memory LoRA (rank 32, synthetic-trained adapter) | 160 | 50 (job 6) | identical |

* **No state leakage:** stock after 46/6 + budget memory + LoRA reproduces stock's first-job hashes;
  46/6 repeated after the budget job and the stock jobs reproduces itself; the weight fingerprint
  was unchanged after all 7 jobs. The methods do change the output (each differs from stock), so
  the equality is not a degenerate "nothing ran" result. Peak allocated memory per job matches the
  one-shot run (stock 80.2 GB, 46/6 114.8 GB, budget 92.6 GB, LoRA 80.8 GB).
* **Speed.** Mean job 49.6 s warm vs 180.7 s one-shot (3.6x) on an H200; the warm startup (container
  start + 14B load + one warm-up chunk) was about 112 s, paid once. Jobs per slot-hour:
  one-shot 19.9; warm 72.5 steady state, 54.9 including one startup for a 7-job session. The
  one-shot figure is generous to one-shot: `job_seconds` excludes container scheduling and image
  start (tens of seconds more per job in practice). Longer sessions amortize the startup further.
* **GPU-model caveat (found while testing):** Modal may serve an `H100` request on an H200. Our
  first warm "H100" session landed on an H200 while its one-shot references ran on a true H100,
  and their hashes differ, as expected for different GPU models (`compare_runner_results.py` now
  flags this and prints each job's GPU). Bit-exactness claims therefore hold only within one GPU
  model; compare warm and one-shot on the same hardware. Within that session the warm repeat of
  job 1 (job 4) was identical to job 1, so the runner itself was state-clean there too.
  A real H100 80 GB did run stock with the lossless savings (expandable segments + CPU text
  encoder), so stock-sized jobs fit there.

## Usage

```bash
export MODAL_PROFILE=nissan-pow
LINGBOT_RUNNER_MAX_CONTAINERS=2 LINGBOT_RUNNER_SCALEDOWN=600 modal deploy scripts/modal_lingbot_runner.py
python scripts/lingbot_runner_client.py jobs.jsonl --gpu H100 --containers 2 --out results.json
# or, without deploying:
modal run scripts/modal_lingbot_runner.py --jobs jobs.jsonl --gpu H100 --containers 2
modal run scripts/modal_lingbot_runner.py --jobs jobs.jsonl --mode oneshot --gpu H100   # reference
```

(The client needs a Python that has `modal`, e.g. `~/.local/share/uv/tools/modal/bin/python`.)

`jobs.jsonl`, one JSON object per line (full spec in `worldsync/lingbot_runner_core.py`):

```json
{"id": "stock-GC007", "path": "turn_return", "task_id": "GC007", "seed": 42}
{"id": "ctx46", "path": "turn_hold_return", "cache": 46, "method": "stock"}
{"id": "budget", "path": "turn_hold_return", "method": "budget10native"}
{"id": "raw", "argv": ["--split", "gc", "--task-id", "GC001", "--max-chunks", "2", "--native-fixed-controls", "--fixed-actions", "WAIT", "W"]}
```

Methods: `stock`, `pose`, `pose-native`, `reanchor`, `decoprune`, `kv<N>`, `budget<K>[native]`;
anything else goes in `extra` as raw rollout flags (`--lora PATH` etc. need `--controlled-ablation`
in `extra` as before). Results land in the `worldsync-lingbot-results` volume exactly like the
one-shot path (`<id>-<time_ns>/run.json`, chunks, controls) and the call returns the report plus
`chunk_pixel_sha256`, `job_seconds`, `weights_unchanged`.

GPU choice: **H200** for memory-heavy methods (46/6 peaks about 115 GB, budget memory 93-98 GB,
KV retention 97 GB). **H100 80 GB** only for stock-sized runs; H100 jobs default to the lossless
savings (expandable-segments allocator, text encoder offloaded to CPU after encoding). RTX PRO 6000
does not work: the image's PyTorch 2.6 lacks sm_120.

`--containers N` is the number of concurrent jobs, hence GPU containers (the class also has a
hard `max_containers`). Idle warm containers hold a slot until `scaledown_window` (default 600 s),
so run `modal app stop worldsync-lingbot-runner` when finished if capacity is tight.

Known limits: the image bakes the PlayWorld GC images only (other splits need the cohort image);
adaptive episodes (player API calls) are not wired into the runner; a CUDA fault (not an OOM)
poisons the process, so restart the container (`modal app stop`) if a job reports one.
