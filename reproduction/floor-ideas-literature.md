# Literature search: ideas for lifting floor episodes (2026-11-15, ~03:20-03:50 PST)

Read-only research, no GPU. Scope: ideas that can be implemented and tested in about 3 hours on
LingBot-World2 14B causal-fast (distilled 4-step causal DiT, KV cache, Plücker camera conditioning,
per-chunk prompt switching already working) with no training.

Problem (full 171-case run, stock 1.85): out-of-sight has 33/43 episodes at exactly 1.0 and insight 16/29,
mostly through the rubric's "no meaningful evolution" cap; IF has 18/50 instruction-gate failures.
KV-memory retrieval gave no judged gain.

Legend: **[V]** = verified by reading the primary source (paper text, abstract, or code);
**[I]** = my inference, not stated by the source.

## Target arithmetic [V]

PlayWorld Table 2 (arXiv 2608.13552v2 HTML), per dimension GC / IF / insight / out-of-sight / overall:

| Model | GC | IF | Insight | Out-of-sight | Overall |
|---|---:|---:|---:|---:|---:|
| Genie 3 | 2.74 | 2.40 | 1.51 | 1.81 | 2.12 |
| LingBot-World2 (hosted) | 2.04 | 2.13 | 1.95 | 1.16 | 1.82 |
| Ours, stock, local | 2.41 | 1.84 | 1.68 | 1.47 | 1.85 |

Beating 2.12 needs the four means to sum above 8.48; ours sum to 7.40. Genie 3's lead is GC and IF, not insight.
PlayWorld Table 3 gives trajectory-validity pass rates for hosted LingBot-World2: GC 58.3%, IF 85.1%, out-of-sight 93.0%.
Our stock IF gate passes 23/50 (46%), so **IF subject response is where we are clearly below the paper** [V numbers, I conclusion].
The paper itself says persistent state evolution is the central unsolved problem: during observation "an ongoing process may
remain static, reset, or evolve without preserving its causal order" and after leaving view "a target may reappear unchanged".
No follow-up paper or leaderboard note explaining Genie 3's out-of-sight score was found.

## Ranked top 5 (expected gain x feasibility in 3 hours)

### 1. Event-style chunk prompts in the format the model was trained on  (IF, out-of-sight, insight)

- Source: LingBot-World 2.0 paper (arXiv 2607.07534; local copy `external/LingBot-World2/paper.pdf`), sections 2.3, 3, 4.2, Figs. 6-8.
- Mechanism [V]: training used a stable **background prompt** plus **chunk-wise captions**; each frame's cross-attention sees the
  background prompt and the chunk prompts up to the current chunk. Chunk captions are composed from event tracks
  (subject visibility, motion state, interaction state, environmental dynamics, static scene state), are constrained to visible
  evidence, and exclude future-revealing or speculative text. The hosted "Director" (a VLM) turns user actions into
  "event proposals" that are fed to the video model as prompts; in the hosted UI event keys trigger subject actions.
- Trained prompt style [V], from the paper's figures: one event per chunk, present tense, action plus visible consequence, e.g.
  "The rider dismounts and sprints across the dunes, his boots striking the sand with force. Golden dust plumes erupt around his heels..."
  and "[T] Push the door handle downward while applying forward pressure to swing the door open."
- Stasis cues to avoid [V text, I effect]: the paper's idle template says "Neither the X nor the camera moves on its own ...
  With no event key pressed, the X rests ...". "wait and observe" appended to a caption already cost about 1.0 on insight dev cases.
- What to do [I]: director emits `background caption` + one event caption per chunk naming the subject, its motion and the
  visible accumulated change; for IF, state the subject's motion that corresponds to the commanded key (locally WASD only moves the
  camera pose, while the hosted service moves the subject).
- Needs: nothing new (per-chunk switching exists). The released pipeline takes one context string; concatenating background + current
  event text is the closest emulation of the trained cross-attention layout [I].
- Effort: low. Risk: prompts that describe changes the camera cannot see may cause scene edits (hurts GC); keep GC on plain captions.

### 2. Normalized Attention Guidance (NAG): negative prompts for a CFG-free distilled model  (insight, out-of-sight)

- Source: arXiv 2505.21179 (NeurIPS 2025); code https://github.com/ChenDarYen/ComfyUI-NAG.
- Mechanism [V abstract/README]: few-step distilled models cannot use CFG; NAG restores negative prompting by extrapolating in
  **attention space** (positive minus negative cross-attention output), then L1-normalising and blending. Validated on Wan2.1-T2V-14B
  and distilled video models; maintainers suggest higher `nag_tau`/`nag_alpha` for few-step models.
- Why it fits [V code, I effect]: Wan's default negative prompt already contains "静态 / 静止" (static, still)
  (`wan/configs/shared_config.py`), but the causal-fast path runs without CFG, so that anti-static term is never applied.
- Needs: a second cross-attention evaluation per layer with the negative context (cross-attention only, not a second full forward),
  in `WanCrossAttention.forward` (`wan/modules/model_fast.py`), with its own cached K/V.
- Effort: medium (about 1-2 hours including a scale sweep). Risk: artifacts or over-saturated motion at high scale; untested on a
  causal KV-cached model; changes generation for every dimension, so needs a GC no-harm check.

### 3. KV recache when the chunk prompt changes  (IF, out-of-sight)

- Sources: LongLive arXiv 2509.22622 (KV-recache) [V abstract]; Grounded Forcing 2604.06939 (Asymmetric Proximity Recache) [V abstract];
  SWIFT 2605.09442 (training-free semantic injection cache) [V abstract].
- Mechanism [V]: cached self-attention K/V of past frames were computed under the old prompt and carry its semantics, so a new
  prompt takes effect weakly; recomputing the cache with the new prompt gives "smooth, adherent switches". SWIFT states the
  mismatch explicitly: memory preserves visual continuity while prompt switches demand semantic adaptation.
- What to do [I]: on a prompt change, re-run the clean (t=0) cache pass over the latents in the current window with the new text context.
- Needs: access to the window's clean latents and the clean-pass routine (the pipeline already runs one per chunk).
- Effort: medium. Risk: LongLive trained with recache, we would not; one extra forward over up to 12 frames per switch.

### 4. Shrink or re-time the static sink for long stationary episodes  (insight, out-of-sight)

- Sources: Deep Forcing arXiv 2512.05081 [V abstract]: StreamingLLM-style sinks cause "fidelity degradation and motion stagnation";
  DySink 2605.21028 [V abstract]: static early-frame sinks "bias generation toward outdated cues" and can cause "sink collapse, where
  content regresses toward sink frames".
- Why it fits [I]: we always attend 6 sink frames from the start of the episode; insight scores fall with length (same videos
  trimmed to 15/30/45/60 s scored 1.83/1.60/1.77/1.17).
- What to do: test `--sink-size` 2 (or 0 if the tuning check allows) with the window unchanged, on insight and out-of-sight dev cases.
- Effort: trivial (flags exist). Risk: more drift and identity loss; direction is uncertain because Deep Forcing's own fix is a
  *larger* sink with re-aligned RoPE, not a smaller one.

### 5. Entity pinning in the background prompt  (out-of-sight identity on return)

- Sources: WorldDirector arXiv 2607.02517 [V abstract only]: an LLM plans object trajectories separately from rendering and keeps
  "visual identity of dynamic entities" after long absences (needs their trained model); LingBot background prompt [V].
- What to do [I]: the director keeps a fixed entity sheet (colour, material, count, position) in the background prompt for the whole
  episode and repeats it on the return chunk, while the event caption carries the progressed state.
- Effort: low (part of the director). Risk: text identity is weak compared with visual memory; our KV retrieval already targets
  this and showed no judged gain.

## Considered and ranked lower

- Conditional image leakage, start from an earlier timestep (arXiv 2406.15735) [V abstract]: fixes low motion in I2V models that
  over-rely on the conditioning image. Only the first chunk is image-conditioned here; later chunks condition on the KV cache [I], so
  the effect on long rollouts is doubtful. Effort low, expected gain low.
- OPSD-V (2607.08766), Phased DMD, Salt (2604.03118): fix motion loss from distillation, but all need training. Out of scope.
- MotionEcho (2506.19348): training-free motion customisation for distilled models with a teacher at test time; we have no
  non-distilled teacher checkpoint (causal-pretrain is unreleased). Not usable.
- Genie 3 promptable world events: only secondary coverage found; no mechanism published. Nothing to copy.
- PERSIST, Mallea, Matrix-Game 3.5: architectural or trained state models (already in the repo survey); not usable in 3 hours.

## Notes for whoever implements

- The out-of-sight and insight cap is an upper bound from an "evolution quality level" on the task's named process; the released
  rubric says any verified persistent change earns at least level 2 [V `metrics/vqa/rubric.py`]. So small but real, accumulated
  change is worth more than visual polish. The director must still never be given rubric text.
- Ideas 1 and 5 are one method (the director); 2, 3 and 4 are independent switches that can be ablated on the same dev cases.
- Nothing here was run. All expected effects are untested on this checkpoint.
