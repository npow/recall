# Stock Matrix adapter reconstruction

The published Matrix adapter is absent from the PlayWorld release. We searched
the complete public source history, the project demo page, dataset, leaderboard
Space, and the public adapter discussion. This implementation reconstructs that
missing boundary and preserves the upstream generator. It does not certify
numerical reproduction of the paper.

## Recovered and verified

* The entire interactive inference file is byte-identical to pre-paper commit
  `6b213ca004dce79eecc6938d3213ae9ffc5a5ef8` (March 27).
  [Verification](matrix-io-verification.json) includes its SHA-256.
* The checkpoint's diffusion model, VAE, text encoder and tokenizer weights have
  identical upstream LFS hashes at the April 28 and October revisions.
  [History comparison](matrix-checkpoint-history.json) rules out a later silent
  weight change for those files. Which released variant the paper used is still
  undocumented.
* All nine headless control mappings were compared with the actual upstream
  `get_current_action()` function using simulated keyboard input. Tensor equality
  passed for both keyboard and mouse channels. Camera magnitude is 0.1.
* Native generation uses 57 frames initially, then 40 per chunk. The actual
  upstream export is 17 FPS. A 25-chunk rollout contains 1,017 frames, or 59.82 s.
* The paper's player, `claude-haiku-4-5-20251001`, successfully returned a validated
  reference-guided decision through the direct Anthropic API. No gateway account
  or separate Agent Player API credential is required.

## Fixed reconstruction choices

Use the released distilled model, demo resolution 704×1280, three sampling steps,
shift 5, seed 42, int8 quantization and original MG-LightVAE decoder. Hardware
substitutions are single-GPU serial decoding and PyTorch SDPA on Thor. Compilation
is disabled. These are fixed starting settings, not a parameter search.

The adapter replaces only two I/O callbacks: console controls become validated
agent controls, and the video callback captures generated pixels before the
upstream key/mouse overlay. Latent generation, camera pose computation, memory
retrieval, denoising and decoding are unchanged. END stops between complete
chunks. Every generated chunk, actual action, decision, request usage, failure and
termination reason is recorded. API errors fail the run instead of silently
falling back to another controller. Existing output directories cannot be
overwritten.

The player sees the initial and latest frames, objective, verbatim reference
action plan, and executed history. It does not receive judge questions. Its prompt
and mapping of Keep/Stop/Extend/Correct/End onto native chunks are reconstructed;
the exact unpublished Matrix prompt and segment scheduler remain unverified.
The 25-chunk time cap fits the paper's approximate 60-second horizon and remains
within its 40-interaction budget; the paper's exact Matrix cap is unknown.

## Run

```bash
# Download about 44 GB, reusing local files only after checksum equality.
.venv/bin/python scripts/prepare_matrix.py

# The isolated Python 3.12 environment uses matrix-runtime-requirements.txt
# plus the host's working CUDA torch and existing shared Python dependencies.
.venv-matrix312/bin/python scripts/verify_matrix_io.py

# Transport/generation smoke test: a single neutral native chunk.
.venv-matrix312/bin/python -m worldsync.matrix_baseline \
  --split gc --task-id GC007 --max-chunks 1 --fixed-actions WAIT \
  --output outputs/matrix-reproduction/smoke

# First adaptive stock rollout; Haiku matches the paper's player model.
.venv-matrix312/bin/python -m worldsync.matrix_baseline \
  --split gc --task-id GC007 \
  --output outputs/matrix-reproduction/gc-GC007-seed42
```

`--dry-run` validates the source, case and configuration without loading weights or
calling an API. The full suite requires all 171 cases, with coverage reported per
split. A smoke test or one adaptive trajectory cannot satisfy that requirement.

## Scoring status

The released scorer's third-person gate expects `third-person`, whereas the
dataset stores `third person`. The local Claude adapter now records and normalizes
this metadata spelling. It also skips the trajectory gate for stationary insight
cases, as required by the released dispatcher. Questions and scoring policies
are otherwise preserved. Upstream's seven evaluator unit tests pass unchanged.

The binary paper formula and the released category scorer are distinct protocols;
we have not recovered the original per-case judgments or a full rollout archive
from the searched public artifacts. Claude can run diagnostic scoring with the
released prompts, but cannot recreate Gemini's original judgments. Numerical
reproduction and hillclimbing remain locked. No Gemini calls are authorized or
made.
