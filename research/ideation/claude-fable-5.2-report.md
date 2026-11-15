I'll start by researching the benchmark and the named models.# PlayWorld hackathon: four ranked project ideas for an open world model that holds together

I would build Idea 1 (memory that ages), wrapped in a small version of Idea 4's autonomous loop, with Idea 2 running unattended on spare GPUs. This is from literature and repo research only; nothing was run, so every effect size is my estimate.

## What the benchmark rewards

PlayWorld's overall score is the unweighted mean of four dimension scores, each on a 1–5 scale ([paper](https://arxiv.org/abs/2608.13552), [repo](https://github.com/kxding/PlayWorld), [dataset](https://huggingface.co/datasets/jocelynd/playworld-bench)).

| Dimension | Cases | What it tests | Genie 3 | Best open | Matrix-Game 3.0 |
|---|---|---|---|---|---|
| Geometry consistency (GC) | 48 | Mostly "rotate 360°" or "go and come back" to the first view | 2.74 | 2.14 | 1.30 |
| Interaction fidelity (IF) | 50 | Mostly a third-person character walking into things | 2.40 | 2.23 | 1.25 |
| Insight evolution (IE) | 30 | "Wait and observe" a visible process for up to 60 s | 1.51 | 1.95 | 1.00 |
| Out-of-sight evolution (OE) | 43 | Look away from an ongoing process, then look back | 1.81 | 1.43 | 1.00 |

Three facts drive the ideas:

- **Half the score is about the world changing, not staying the same.** The IE and OE rubrics cap a frozen or copy-pasted video at 1. Matrix-Game 3.0 scores exactly 1.00 on both.
- **Memory and evolution pull against each other.** A static memory returns a stale snapshot, which is precisely the "frozen copy" the OE rubric punishes. STEVO-Bench reports that memory architectures worsen the static-scene bias of camera-controlled models ([paper](https://arxiv.org/abs/2603.13215)).
- **The space is crowded.** Training-free static memory exists (WorldKV, WorldTrace, Closing the Loop, World in World). Out-of-sight evolution exists but only with training (LiveWorld, ReMind, EvoWorld, HyDRA, World Observer, Momentum). I found no training-free method for out-of-sight evolution.

## Idea 1: Memory that ages (training-free out-of-sight evolution)

**Pitch.** Put an "aging" step between memory store and memory retrieval, so static content comes back identical and dynamic content comes back evolved.

**Research question.** Can a frozen open world model gain out-of-sight evolution without losing revisit consistency, purely by changing what its memory returns?

**Mechanism.**
- **Base model:** ABot-World-0, because it is 5B, Apache-2.0, runs on one GPU, and takes keyboard actions, a text prompt and reference-image tokens. It has only a bounded local attention cache and lists "persistent scene memory" as future work ([paper](https://arxiv.org/abs/2607.19191)).
- **Pose ledger:** dead-reckon camera pose from the action stream, calibrated once. Store keyframes and cached attention state by pose, and always pin the first frame.
- **Mutability map:** mark tokens as static or dynamic from latent variance over the first chunks, plus a short hidden "wait" rollout forked from the current state. A vision-language model names the process and logs it as JSON.
- **Aging operators (the ablation axis):**
  - A0: stale snapshot, which is what existing training-free memories do.
  - A1: fork the model's state when the target leaves view and keep rolling it under "wait", catching up only when a return is imminent.
  - A2: re-noise dynamic memory tokens in proportion to elapsed time, so the model re-imagines them near their old state.
  - A3: have an image-edit model render "the same scene N seconds later".
- **Injection:** static tokens come from the old keyframe and dynamic tokens from the aged one. Inject through cache insertion or ABot's reference-token slot. For pure rotations, which are most GC and OE cases, fall back to exact homography compositing and then hand the main rollout over to the aged fork.

**Why it should help.** It targets OE and GC, which are 91 of 171 cases and half the score.
- OE: +0.5 to +1.0.
- GC: +0.3 to +0.6.
- Overall: +0.2 to +0.4.
- The gain is bounded by how well the base model evolves a scene under "wait".

**Novelty check.**
- [LiveWorld](https://arxiv.org/abs/2603.07145) has stationary "monitors" that fast-forward entities, but needs a trained renderer. A1 is that concept done training-free, which I would state openly.
- [EvoWorld](https://arxiv.org/abs/2611.06748) (4 Nov 2026) uses a vision-language "brain" plus a trained renderer. Its own ablation shows the brain's prompt alone does not transfer to LingBot-World, which motivates injecting pixels rather than text.
- [ReMind](https://arxiv.org/abs/2605.25333), [HyDRA](https://arxiv.org/abs/2603.25716), [World Observer](https://arxiv.org/abs/2610.02162) and [Momentum](https://arxiv.org/abs/2610.23423) all train.
- [WorldKV](https://arxiv.org/abs/2605.22718), [WorldTrace](https://arxiv.org/abs/2608.07408), [Closing the Loop](https://arxiv.org/abs/2607.21848) and [World in World](https://arxiv.org/abs/2609.11548) are training-free but return static memory.
- A2 borrows its principle from fractional history guidance ([DFoT](https://arxiv.org/abs/2502.06764)).
- New here: an aging operator in the memory path of a frozen model, the static/dynamic split, and measuring the GC-versus-OE trade-off on PlayWorld.

**Minimal experiment.**
- Baselines: base model, and base plus stale retrieval.
- Ablations: A0 to A3, and mutability gating on or off.
- Metrics: PlayWorld GC, OE and IE with gate pass rates. Add judge-free checks: similarity of the returned view on static regions, and change on dynamic regions.
- Falsifiers: aged retrieval fails to beat stale retrieval on held-out OE by at least 0.3, or it costs more GC than it gains OE. If stale memory does not lower OE relative to no memory, the premise is wrong.

**Autonomous loop.** An agent runs the ablation grid, reads the verifier's per-question evidence, and learns which aging operator to use per process type as a bandit over dev results.

**One-day plan.**
1. Hours 0–3: write the adapter and baseline 40 dev cases.
2. Hours 3–8: pose ledger and fork-and-handoff for rotation-only cases.
3. Hours 8–14: cache or reference-slot injection, then A2 and A3.
4. Hours 14–24: full run, ablations, held-out scoring.

**Risks.**
- ABot may not evolve processes under "wait". In that case use ReMind-5B or the Wan2.2 base as the evolver, since they share its latent space.
- The reference slot was trained for character identity and may not act as scene memory.
- A visible pop at the handoff could read as an artifact to the judge.

## Idea 2: Merging sibling world models in weight space

**Pitch.** Several open world models are fine-tunes of the same backbone, so search for a weight merge that combines their skills with no training and no harness.

**Research question.** Do world-model capabilities such as keyboard control, liveness and out-of-sight memory compose in weight space?

**Mechanism.**
- **Siblings:** Matrix-Game 3.0, ABot-World-0, ReMind-5B, Yume-1.5 and WorldPlay-5B all derive from Wan2.2-TI2V-5B ([MG3](https://huggingface.co/Skywork/Matrix-Game-3.0), [ReMind](https://huggingface.co/AppliedIntuitionResearch/ReMind), [Yume](https://huggingface.co/ckurasek/Yume-1.5-5B-720P-MLX), [WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay)).
- **Why ReMind matters:** it is the strongest model on STEVO-Bench state progress (14.5, against 2.9 for Genie 3 and 3.4 for LingBot-World).
- **Merge:** take each model's difference from the base as a task vector, and add scaled vectors onto ABot's weights per block group, keeping ABot's action adapter.
- **Search:** evolutionary search (CMA-ES) over 20–60 coefficients on 8 GPUs.
- **Fitness (judge-free):** action following, loop closure, look-away progress on non-benchmark images, and a quality floor.

**Why it should help.** A successful merge would lift OE and IE inside a single real-time model. I put the chance of a merge that keeps control and visibly gains evolution at 20–30%; if it works, OE could rise by 0.3 to 0.8.

**Novelty check.**
- [Task arithmetic](https://arxiv.org/abs/2212.04089) and [evolutionary model merging](https://arxiv.org/abs/2403.13187) are established for language and image models.
- NVIDIA's Cosmos reportedly merges its own domain specialists ([secondary source](https://ceyxasm.github.io/posts/cosmos-wfm-training/)).
- [Composition of Memory Experts](https://arxiv.org/abs/2605.18813) composes experts at the score level, not in weights.
- [EvoPhysWM](https://github.com/srt-tkyk/EvoPhysWM) is a toy PDE version.
- I found nothing merging independently released interactive video world models with different control interfaces. This is the most novel idea here.

**Minimal experiment.**
- Baselines: each parent, a uniform average, and random coefficients at equal budget.
- Then score the best merge on PlayWorld.
- Falsifier: no merge beats both parents on any two fitness terms without dropping the third below its floor.

**Autonomous loop.** The project is the loop: evolution improves from its own results, every recipe is a logged vector, and the per-block coefficient map shows where each capability lives.

**One-day plan.**
1. Hours 0–4: loaders and key matching.
2. Hours 4–8: fitness suite.
3. Hours 8–20: search.
4. Hours 20–24: PlayWorld scoring.

**Risks.**
- ABot is a distilled few-step model while ReMind's few-step behaviour lives in a separate LoRA, so merges may break sampling.
- ReMind's camera-phase attention modules have no counterpart in ABot.
- A clean negative result is the most likely outcome.
- Check the ReMind and WorldPlay weight licences.

## Idea 3: Dream, repair, distill

**Pitch.** Use the slow harness from Idea 1 as a teacher that produces better rollouts than the model alone, distill them into a LoRA, and repeat.

**Research question.** Can a world model improve its own long-horizon consistency from its own repaired rollouts, including for irreversible processes?

**Mechanism.**
- An explorer agent generates loops, look-aways and waits on non-benchmark images.
- The harness repairs each rollout with an anchored return and aged memory.
- Judge-free checks and a cheap vision-language model filter the results.
- Train a rank 64–128 LoRA on the repaired segments only, with a term that preserves the original outputs elsewhere.
- Round two runs the improved model inside the harness.

**Why it should help.** The gain survives without the harness and in real time. I would expect round one to recover 30–60% of the harness gain.

**Novelty check.**
- [WorldCycle](https://arxiv.org/abs/2608.04964) supervises with reversible action cycles. That signal cannot see irreversible evolution, and its closure reward would penalise it.
- [ReWorld](https://arxiv.org/abs/2608.23565) trains on palindrome trajectories.
- [WoRL](https://huggingface.co/qyoo/lingbot-worl) is a reinforcement-learning LoRA for LingBot-World revisits.
- [World Model Self-Distillation](https://arxiv.org/abs/2606.12072) distills task solving, not consistency.
- New here: the teacher is the model plus a test-time harness, so supervision covers evolving processes.

**Minimal experiment.**
- Conditions: base, harness, distilled without harness, distilled with harness.
- Ablation: train on loops only versus loops plus look-aways. I predict loops-only lowers IE and OE.
- Falsifier: the distilled model without the harness shows no held-out gain, or its liveness drops.

**Autonomous loop.** Explore, repair, filter, train and re-evaluate with no human step; each round's model generates the next round's data.

**One-day plan.** Only viable if Idea 1 works by midday: generate 500–1,000 clips in the afternoon, train for 3–4 hours, evaluate.

**Risks.**
- A LoRA on a distilled student may blur few-step output.
- The model may learn harness artifacts.
- No public ABot training code was found (the repo documents inference and a dataset), so the loss must be written.

## Idea 4: Self-evolving harness

**Pitch.** A research agent owns the harness, reads the verifier's written evidence for each failed question, proposes one patch at a time, and keeps it only if a validation split improves.

**Research question.** How much PlayWorld score is recoverable by automated harness search alone, and does it transfer to held-out cases and a second judge?

**Mechanism.**
- **Search space:** degrees and frames per step, memory thresholds, anchoring strength, aging operator, prompt schedule for IE, sampler steps, and small code patches.
- **Guardrails:** a dev/validation/test split, revert on regression, two scoring passes, judge-free metrics as a second objective, and a second verifier model. The system never sees rubric text at inference.

**Why it should help.** Much of GC is lost at the trajectory gate rather than in the rubric. Matrix-Game 3.0 passes it 37.5% of the time and Genie 3 77.1%, so calibration alone is worth points. Estimate: +0.1 to +0.3 overall.

**Novelty check.**
- [EvolveMem](https://arxiv.org/abs/2605.13941) runs the same loop for language-agent memory.
- [Harness²](https://arxiv.org/abs/2610.14224) evolves agent harnesses.
- [LingBot-World 2.0](https://arxiv.org/abs/2607.07534) ships a hand-designed pilot and director harness.
- Applying the loop to a world-model harness is a new use case, not a new method. This is the least novel idea but the strongest on autonomy.

**Minimal experiment.**
- Baselines: a hand-tuned harness, and random search at equal budget.
- Falsifier: validation gains do not transfer to the test split or the second judge, or random search matches the agent.

**Autonomous loop.** The agent proposes, runs, scores and reverts unattended, and a logged library of accepted and rejected patches feeds its next proposals.

**One-day plan.** The easiest to stand up: loop skeleton by hour 6, then it runs.

**Risks.** Overfitting the judge, verifier API cost, and looking like benchmark gaming unless the split is pre-registered.

## What I would pick and what to know first

I would pick Idea 1 because it attacks the dimensions with the most headroom and degrades gracefully: the rotation-only fork-and-handoff tier needs no access to model internals beyond saving and restoring state. Use Idea 4's loop to run its ablations. Launch Idea 2 on two or three GPUs as an unattended side bet. I would not promise 2.12: my estimate is a 25–35% chance of passing Genie 3 in a day.

Facts to know before starting:

- **Protocol.**
  - The agent player is Claude Haiku 4.5 with a 40-step budget.
  - Each action step is 450 ms, and the action strings imply 22.5° per rotation step (16 steps for 360°).
  - The verifier is Gemini 3.1 Pro reading contact sheets at 10 fps (384×216) and 0.5 fps (800×450).
  - Scoring is a single pass, with a reported two-pass variance of 0.0112.
- **No adapter is provided for your model.** The repo ships adapters for Genie 3, HappyOyster, HY-World 2 and SANA-WM only. You write the action-to-control mapping, and it largely determines the gate pass rate.
- **ABot-World-0 has no PlayWorld score.** Baselining it is the first job and is itself a new data point.
- **Matrix-Game 3.0 is a poor base for beating Genie 3.** With IE stuck at 1.00, the other three dimensions would need to average about 2.5.
- **LingBot-World 2.0 is the best open model at 1.82.**
  - It was scored through its web interface, so local weights may not reproduce that number.
  - The 14B reference script uses 8 GPUs at 480p.
  - The licence is CC BY-NC-SA.
- **Latency is not scored.** Slow harnesses and extra sampling steps are free.
- **A shortcut exists.** [WorldKV's code](https://github.com/cvlab-kaist/WorldKV) already implements training-free retrieval on LingBot-World-Fast, if you prefer that base.
- **Leaderboard.** I could not render the live leaderboard page; the numbers above are the paper's Table 2, which the repo mirrors.
