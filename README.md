# WorldSync

Training-free visual-memory experiments on **ABot-World-0-5B-LF**.

This implements a **WorldTrace-Landmark-inspired adaptation**, not the authors'
implementation or a reproduction of their published results. The published
WorldTrace experiments primarily use Matrix-Game 2. ABot already stores raw
pre-RoPE keys and assigns bounded relative positions; our change retains selected
older frames within that existing attention window.

## Experiment

Three modes share the checkpoint, input image, prompt, actions, random seed,
resolution, unquantized BF16 weights, decoder, and SDPA backend:

* `stock`: unchanged upstream cache policy, with its normal fast relative RoPE path.
* `window`: our cache adapter with FIFO selection, retaining 21 recent latent frames.
* `landmark`: our adapter with 15 recent latent frames and up to 6 older scene-entry
  frames. Scene entries use cosine distance between sampled raw key features.

`window` checks whether differences arise from adapter mechanics. Its output is
tested against upstream's non-fast relative attention after cache eviction.
Small numerical differences from stock's fast path can still compound during
generation. Compare `landmark` against both controls.

On each denoising step, the current block replaces pending K/V. It is committed
only when the next block begins, after upstream has overwritten it with the clean
context. Selected raw keys stay frozen and upstream applies spatial and temporal
RoPE once at read time. Frame slots receive distinct bounded positions; the query
occupies the tail. Reference-image tokens remain separate and unchanged.

The adaptation has no field summaries, attention-response merge scoring, object
state simulator, geometry correction, or past.dev dependency. Long generation
alone does not demonstrate persistent physical state. Older visual memories can
also restore stale states, which must be checked in dynamic-scene experiments.

## Setup

Linux, Python 3.12, CUDA-enabled PyTorch and Triton are needed for ABot. This machine
uses NVIDIA Thor (ARM64) and PyTorch 2.15.0+cu132. Its local `.venv` borrows the
existing CUDA installation through `existing-cuda.pth`; no packages in that other
environment are modified. On another machine, install a matching CUDA build of
PyTorch/Triton into your own environment first, then:

```bash
uv venv .venv
# Install your CUDA PyTorch/Triton build here before resolving the remaining deps.
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python scripts/setup.py
.venv/bin/python scripts/download_weights.py
.venv/bin/python -m pytest -q
```

The upstream checkout is pinned to `5c3f7260f2346ebb60a49efb81a7db5bfe15d7e5`.
The patch is opt-in and leaves normal ABot behavior intact until enabled. External
code, checkpoints (about 25GB), environments and generated outputs are ignored by Git.
Upstream's Apache-2.0 license and checkpoint notices remain in their download locations.

## Run

```bash
# One-block smoke test, explicitly marked as an incomplete action sequence.
.venv/bin/python -m worldsync.run --max-blocks 1 --output outputs/smoke

# Paired pan-and-return diagnostic. Each output directory must be new.
.venv/bin/python -m worldsync.run --seeds 42 43 44 --output outputs/pan_return
.venv/bin/python -m worldsync.score outputs/pan_return
```

Default resolution is 352×640 for a manageable first experiment, below the release's
720p reference setting. Add `--height 704 --width 1280` to test that setting. The
pan trajectory uses equal left/right action durations; successful return is **not
guaranteed**. Inspect the video before interpreting first/last frame similarity.
The novelty threshold of 0.15 is an initial uncalibrated setting, not a published
ABot optimum. Tune it on separate development cases before held-out evaluation.

Outputs include videos, endpoint frames, per-block actions/times, layer-zero memory
IDs, upstream/checkpoint revision, seed, GPU memory and environment details. The
DINOv2 score is an appearance diagnostic, **not a PlayWorld score**. A frozen video
can score well while completely failing the intended movement.

## PlayWorld evaluation

The repo includes deterministic open-loop diagnostics and a native adaptive player.
The player defaults to **GPT-6.1 Sol**, with **Claude Opus 5.5** available explicitly.
There is no automatic downgrade. The benchmark's published player uses Claude
Haiku, so our controller is a protocol change that must be reported. Do not compare
DINO cosine values against the published 1–5 PlayWorld table.

```bash
.venv/bin/python scripts/setup.py --with-playworld
.venv/bin/python -m worldsync.agent_run \
  --dataset external/PlayWorld/example/GC/002.json \
  --data-root external/PlayWorld/example --task-id GC004 \
  --output outputs/adaptive-gpt6

# Explicit alternate provider; uses ANTHROPIC_API_KEY.
# Add: --provider anthropic --player-model claude-opus-5-5
```

The player receives the scene, objective, observations and action history, without
judge rubrics. Only the image caption is passed to the world generator. Actions
are bounded and validated. Each method resets world memory and decoder state;
the agent model, prompt and budgets stay fixed across compared methods. The
40-decision and approximately 60-second generation limits are recorded, along
with the actual requested and returned API model IDs. A smaller pilot budget is
explicitly recorded if supplied on the command line.

`scripts/playworld_score.py` now uses **Claude Opus 5.5** with PlayWorld's rubric
prompts and dual-scale sampling from pinned revision
`eeaea12e050a90b815d11d030e943e7b3181da21`. No Gemini API is used. It reads
`ANTHROPIC_API_KEY` from the environment or the ignored, permission-restricted
`.env.local`. The default `both` mode validates the trajectory before scoring;
failed trajectories receive 1 and skip the extra rubric call. Actual judge model
IDs, prompt hashes and token usage are saved. This is a provider adaptation, not
the published Gemini evaluation protocol. For a video generated from its matching
benchmark case:

```bash
.venv/bin/python scripts/playworld_score.py --help
.venv/bin/python scripts/playworld_score.py \
  --dataset external/PlayWorld/example/GC/002.json --task-id GC004 \
  --video outputs/adaptive-gpt6-pilot/seed-42/stock/GC004/rollout.mp4 \
  --reference-image external/PlayWorld/example/GC/002.jpg \
  --output outputs/adaptive-gpt6-pilot/stock-judge.json
# Add --dry-run to inspect evidence preparation without an API call.
```

The judge assesses outputs; a separate Agent Player chooses actions while the
world runs. API variables are read from the environment or ignored `.env.local`.
They are never included in result files.

For paper-comparable evaluation, match its Agent Player, judge, budgets and
preprocessing, rather than treating the custom player as an exact reproduction. Report the full
four-dimension score, trajectory pass rates, several seeds, failure clips, and
added latency/memory. Keep memories isolated per episode and expose no judge
rubrics or expected outcomes to the generation policy.

## Research sources

* [WorldTrace paper](https://arxiv.org/abs/2608.07408),
  [method details](https://research.nvidia.com/labs/sil/projects/WorldTrace/).
* [ABot-World release](https://github.com/amap-cvlab/ABot-World).
* [PlayWorld benchmark and custom-model interface](https://github.com/hku-sail/PlayWorld).
* [WorldMem](https://arxiv.org/abs/2504.12369).

No benchmark gain is claimed without measured results.

Initial pilot: three seeds on one pan-and-return scene produced mean DINO cosine
0.3711 (stock), 0.4092 (window control), and 0.3048 (landmark). In the inspected
seed-42 clips, none returned to the original view. This does **not** establish a
memory improvement; the fixed action trajectory fails its intended purpose.
The adaptive GPT-6.1 player experiment addresses that diagnostic failure next.
Local videos and full per-run measurements are under `outputs/pan_return/`.
