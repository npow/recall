# How the memory methods work, and what the evidence says

Track 4 question: how long can a generated world hold together while an agent plays in it?
Base model: **LingBot-World2 14B causal-fast** (open, Wan2.2-based, distilled 4-step causal video DiT with a KV cache).
No training from scratch. Numbers live in [LEADERBOARD.md](../LEADERBOARD.md); this note explains the mechanisms.
Nothing here is a full-benchmark PlayWorld score unless the leaderboard says so.

## The failure we attack

Stock inference keeps a rolling window of 18 latent frames (6 "sink" anchor frames + 12 recent), about 4.5 s of video.
When the camera turns away and comes back after longer than that, the view it left has been evicted from the KV cache,
so the model invents a plausible but different scene (a different courtyard, a re-lettered sign, a missing building).
On a scripted turn-away / hold / turn-back path the stock model returns to a different place
(DINO similarity of the final frame to the start view 0.30-0.40; see [revisit matrix](../reproduction/revisit/README.md)).
A larger cache does fix it (46/6 window: 0.89) but costs 115 GB and 1.45x time, so it is a diagnostic, not a method.

## Method 1: pose-indexed historical KV retrieval (training-free)

Idea: keep the same 18-frame attention budget, but let some of those slots hold *old* frames chosen by camera pose.

1. After each chunk is generated, archive the clean per-layer key/value tensors of its latent frames on CPU
   together with the commanded camera pose (from the pose tracker in `worldsync/lingbot_controls.py`).
2. Before generating a chunk, compare its target pose (position + heading) with archived poses and pick the best matches.
3. Replace part of the attended window with those archived frames and generate as usual. Weights, sampler and schedule are unchanged.

Variants (all default-off flags in `worldsync/lingbot_rollout.py`):
- original four-view retrieval (`worldsync/lingbot_retrieval.py`): validated in
  [memory-validation](../reproduction/memory-validation/README.md) and the
  [task pilot](../reproduction/task-memory-benchmark/README.md) (local Claude rubric 2.47 -> 3.57 on three scripted tasks).
- **`--budget-memory`** (`worldsync/lingbot_budgetmem.py`, our fixed version): the archive is bounded (24 frames, evicting
  pose-redundant ones, about 31 GB on CPU), the attended window is sink 6 + K retrieved + tail 8 (K=10, native RoPE positions),
  and selection uses commanded pose, not a content query from the noisiest denoising step.
  An independent audit of an earlier content-query version found three real bugs
  (blind first-step query, hard-coded 18/6 gate, unbounded archive); this version fixes them
  ([budgetmem](../reproduction/budgetmem/README.md), audit notes in [INTEGRATION.md](../reproduction/INTEGRATION.md)).

Why pose and not content: retrieval has to find the frame that matches the view about to be generated, and at the
first (noisiest) denoising step the query carries no information about the requested camera.

Cost: peak GPU about 93 GB (K=10), about 1.5x per-chunk time versus stock in the warm runner; CPU archive up to about 31 GB.

## What the evidence shows

Fixed revisit paths (GC007, DINO of final frame vs start view; crude, single seed, static scene):

| path | stock | budget memory K10 | 46/6 cache |
|---|---:|---:|---:|
| turn away and back | 0.58-0.70 | 0.77 | 0.82 |
| long hold (24 chunks away) | 0.30-0.40 | 0.77 | 0.88-0.90 |

Judged episodes, Gemini (released PlayWorld prompts, rubric, gate, aggregation; Vertex `gemini-3.1-pro-preview`),
14 GC and out-of-sight development cases (not hash-selected: 7 from the first development cohort, 7 chosen for long reference-plan duration),
run with `--end-guard`, two seeds, paired against stock
([powered comparison](../reproduction/powered-comparison/pooled/README.md)):
stock 1.81, budget memory 2.13 (+0.32, 95% CI -0.08 to +0.75), 46/6 cache 2.26 (+0.46, CI -0.22 to +1.15).
**No arm is established better than stock overall**; budget memory is positive in both seeds and halves instruction-gate failures (3 vs 7 of 28).
These cover two of the four dimensions and are not comparable with Genie 3's 2.12.

**Full benchmark (official).** All 171 cases, both arms, one seed, released Gemini scorer, 341 of 342 episodes scored
([table](../reproduction/full-run/table.md)): stock **1.85** [1.67, 2.04], budget memory **1.80** [1.63, 1.98];
paired budget minus stock **-0.04** [-0.26, +0.16]. Per dimension (stock / budget): GC 2.41 / 2.37, IF 1.84 / 1.76,
in-sight 1.68 / 1.72, out-of-sight 1.47 / 1.36. In-sight uses caption-only text (pre-declared amendment). An independent re-judging
of the same videos bounds stock at 1.76-1.80 and budget at 1.71-1.75 (possible ranges, with two judgments still missing), so single-call judge noise is of about that size.
**Why memory gave no gain** ([memory audit](../reproduction/full-run-memory-audit.md)): the mechanism ran as coded and its fixed-path gain
reproduces, but only about 12% of GC and out-of-sight chunks are real revisits, returns made by translation are not undone by it,
and the 24-frame archive behaves like a roughly 6-second first-in-first-out buffer on distinct poses.
The pre-registered pass criterion for beating 2.12 is **not met**. Memory did fire in 136 of 171 budget episodes, the +0.32 above does not
replicate on its own 14 cases inside the full run (+0.04), and the judge scored 5 of 28 identical-pixel video pairs differently (by up to 4.0 points),
so single-call differences under about 0.3 are noise ([exploratory analysis](../reproduction/full-run-memory-fired.md)).

### Director (text state tracking)

**What it is.** `--director static` (default off, `worldsync/lingbot_director.py`): Claude Haiku writes one event description per case
in LingBot's trained caption format, clean-room from the image caption, task objective and reference plan only (never rubric text).
It replaces the world-model text on IF, in-sight and out-of-sight cases; GC keeps the stock text.
**Development result** ([director-dev](../reproduction/director-dev.md)): on 20 development cases with fixed-action replays and three judge calls,
no reliable gain: out-of-sight +0.33 [+0.03, +0.67] (n=6), in-sight +0.43 [-0.08, +0.98], IF +0.03, about +0.1 to +0.2 overall at best.
**Final run** ([final1](../reproduction/final-run/table.md), Lambda H100, 171 cases, seed 42, single Gemini call):
stock 1.81 [1.64, 1.99], director 1.82 [1.66, 2.01], paired +0.02 [-0.15, +0.19]; held-out 151 cases 1.76 vs 1.82, +0.06 [-0.13, +0.26].
A null; the pre-registered projection was about 2.05, and the pass criterion is not met.
**Gate counts.** IF gate failures fell from 20 to 11 while out-of-sight gate failures rose from 7 to 13, and scores did not move.
Why the two moved in opposite directions has not been investigated.

## Things we tried that did not help (kept in the repo)

- Memory LoRA fine-tune on 48 real Sekai revisit pairs: no gain over plain retrieval and it degrades ordinary generation ([lora](../reproduction/lora/README.md)).
- Start-image re-anchoring: keeps sign identity, not geometry. DeCoPrune and WorldKV-style retention ports: no gain on our paths.
- Larger caches with fp8 KV: 46/6 fits in about 87 GB but quality is mixed (results on the `fp8-kv` fork branch, not yet merged).
- Prompt-based motion tricks, translation gain, measured-motion feedback to the player, turn-rate compensation: within noise.
- Keyframe restart (regenerate from a stored frame): returns to the start view by construction (DINO 0.95-0.99) at stock memory, but shows hard cuts and
  discards anything that moved; untested in adaptive episodes (results on the `keyframe-restart` fork branch, not yet merged).

## Where the world still breaks (see [oe-insight-caps](../reproduction/oe-insight-caps.md), [score-drivers](../reproduction/score-drivers.md))

- Identity loss on return (signs, objects vanishing) is the largest out-of-sight failure, and memory only partly fixes it.
- Frozen or wrong dynamics (a train that stops), physics glitches and long forward runs into walls (smeared texture): no cache fixes these.
- About a quarter of out-of-sight episodes sit at exactly 1.0 after an instruction-gate failure even though the commanded motion happened;
  the judges disagree (Gemini vs Opus per-episode correlation 0.3-0.7), so part of the score is judge noise.
- Interaction (IF) tasks that need non-camera actions, and insight tasks (static "wait and observe") are driven by the world model's own dynamics and the text prompt, not by memory.
- Memory only helps when the camera is away longer than the 4.5 s window; most reference plans are shorter.
- Camera control is mostly realized (forward median 1.78x, backward 0.49x, turns about 177 deg of 180 across 13 scenes), but weak on distant scenes ([near-field](../reproduction/near-field-motion.md)).

## Reproduce

Pre-registered criterion for "beat 2.12": [pass-criterion.md](../reproduction/pass-criterion.md). The full-run config is frozen in a committed file before results (see the leaderboard when the run completes).
Fixed-path diagnostics run on the warm Modal runner ([modal-runner](../reproduction/modal-runner.md)); adaptive episodes need `--experimental-adaptive-ablation`
with `--budget-memory --budget-rope native --budget-retrieved 10`.
Caveats that apply to every number: local checkpoint vs the hosted model in the paper, our player (Haiku with retries and, where stated, an end guard) vs theirs, Gemini preview model name, case subsets, few seeds.
