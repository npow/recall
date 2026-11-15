# DeCoPrune: exact spec read from the released code

Source read: `github.com/DeCoPrune/CMBench` (shallow clone, HEAD at read time), files
`configs/decoprune.json`, `src/cmbench_rebuild/methods/{consistency_prune,consistency,selectors,spec}.py`,
`src/cmbench_rebuild/runtime/{online_policies,generation}.py`, `src/cmbench_rebuild/core/rope.py`,
`src/cmbench_rebuild/config.py`, and the README. The paper (arXiv 2609.39096) was not fetched; everything
below is from code, which is authoritative for the port.

## What is scored

* **Unit:** one KV token = one 2x2 spatial patch of one latent frame (LingBot: 30x52 = 1560 tokens per
  latent frame, 4 latent frames = 6240 tokens per chunk). Tokens are ordered `(frame, h, w)`.
* **Score (`score_type="mse"`):** mean over the 16 latent channels and the 2x2 pixels of
  `(x0_pred - x0_clean)^2`. `x0_pred` is the generator's own clean prediction at one intermediate
  denoising step; `x0_clean` is the chunk's final clean latent (the value written to the cache).
* **Which step (`generation_step_index=2`, zero-based):** third of four denoising calls. Under the
  CMBench schedule that is timestep 899. During generation the score reuses the prediction that
  denoising already computed (`capture_generation_prediction` stores the converted x0 at that step).
  No extra forward pass. For an observed prefix it re-noises the clean chunk at `step_index` and runs
  with and without predecessor KV (`context_probe`); that path is not needed for from-image generation.
* **Which chunks:** chunk 0 is never scored (`context_probe` returns `None` for `chunk_idx == 0`),
  so it stays in the cache whole. Every later chunk gets a record when its clean replay finishes.

## What is pruned and kept

* **Rule:** keep a token iff `score > threshold` (`threshold = 0.10`); prune it otherwise. Non-finite
  score means keep. The diagnostic spatial floor (`min_keep_per_spatial_block`) defaults to 0 = off.
* **When:** `before_clean(chunk m)` runs `_compact_consistency(next = m + 1)`, which applies every
  unapplied record with `chunk_idx < next - 2*local_window_chunks = m - 1` (`local_window_chunks = 1`),
  i.e. chunks `<= m-2` are pruned before chunk `m`'s clean replay. Chunk `m`'s denoising therefore sees
  chunks `<= m-3` pruned and chunks `m-2`, `m-1` complete. Two full recent chunks = 8 latent frames.
* **Scope:** the same mask for every layer and head (`decoprune`; `decoprune_hs` is per-head and is not
  ported). Pruning is physical removal from the cache, permanent.
* **Budget:** none. The cache is unbounded (`max_attention_size` large, no sink eviction). Kept tokens
  persist forever. Reported result: 85.4% of historical KV pruned at 0.6701 DINO on CMBench.

## RoPE re-indexing (`rope_reindex.mode = "all_bands"`)

Default plan `virtual_span=17, recent_frames=8`. With `q0` = number of latent frames before the current
chunk and `offset = q0 - real`, a historical frame keeps its true position if `offset <= 8`; otherwise
`virtual_offset = 8 + (offset-8) * (17-8)/(q0-8)` and position `q0 - virtual_offset`. All older history is
squeezed into a 9-frame virtual span that precedes the 8 recent frames, in **all** temporal RoPE bands
(`fast_bands` would touch only the first 11 pairs). Positions are fractional. Spatial RoPE is unchanged.

## Denoising schedule

CMBench runs the **FlowUniPC four-step schedule with shift 5**: timesteps `(999, 957, 899, 702)`,
`timesteps_index = (0, 179, 358, 679)`, 832x480, 16 FPS, 4 latent frames per chunk, plus the same
KV-cache causal-fast model. Our stock rollout instead uses `shift=10, timesteps_index=[0,250,500,750]`
(what `generate()` is called with in `worldsync/lingbot_rollout.py`). The method is defined relative to the
schedule it was tuned on, so the port exposes the schedule as its own flag.

## LingBot v2 specifics

* Config names `lingbot-world-v2-14b-causal-fast`, our checkpoint (revision pinned in
  `reproduction/lingbot-weights.json`).
* LingBot caches keys *after* RoPE at absolute temporal positions. The port stores canonical
  (post-`norm_k`, pre-RoPE) keys and re-applies RoPE at the target positions on read.
* Stock attends sink 6 + last 8 frames + current 4. DeCoPrune attends: chunk 0 whole + kept tokens of
  chunks `1..n-3` + chunks `n-2`, `n-1` whole + current chunk.

## Where this port deviates from the released code

1. **No context-video prefix.** Round trips generate from a single image, so every scored chunk takes the
   generation-phase path (live x0 at step 2). The observed-prefix re-noise probe is not implemented.
2. **Chunk 0 contains the image latent** and is never pruned (matches "chunk 0 is never scored").
   Under stock, sink = frames 0-5; the port has no sink beyond chunk 0.
3. **RoPE guard:** for `q0 <= virtual_span (17)` positions are left real. The released formula expands
   instead of compressing when `8 < q0 < 17` and yields negative positions (e.g. q0=12 -> -5).
4. **Attention is rebuilt per layer from our own archive** rather than from a physically compacted
   cache. The stock cache is still written (and still evicts) but is not attended. Numerically the attended
   token multiset and positions are what DeCoPrune specifies.
5. **Threshold not re-tuned.** 0.10 was tuned on their setup; we report the realised pruning ratio.
6. **Not ported:** DeCoPrune-HS, the spatial floor, `cosine`/normalized scores, the ForcingKV/Dummy baselines.
7. Single seed, single path. Nothing here reproduces the paper's CMBench numbers.
