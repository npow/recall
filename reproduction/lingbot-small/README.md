This is an independent small-model speed/quality probe, not a lossless 14B optimization or a leaderboard result. The official release name is 1.3B, but strict loading verifies 1,709,502,016 actual DiT parameters. Its six original FP32 shards occupy 6.838GB and are converted to native BF16 for inference. Every shard matches the published HF SHA256 at the pinned revision in weights.json.

The isolated wrapper selects the official i2v-1.3B config (30 layers,12 heads,dim1536), source 1895d300, four ODE indices [0,250,500,750], shift 5. It reuses the previously verified 14B T5/VAE/tokenizer assets and records that asset revision separately from the small DiT checkpoint. No core model/rollout default is changed.

Run the complete 18/6 baseline before the 70/6 variant: GC027, seed 42, WAIT followed by 12 RIGHT commands at 30 degrees/second,13 chunks. Both use the one-shot text-encoder disposal and fused DiT SDPA tested on Thor, with a 16GiB physical reserve. The 70-frame native context changes the experiment; it is not an exact replacement for the short-context baseline.

```sh
.venv-lingbot/bin/python scripts/stage_lingbot_small.py
python scripts/guard_lingbot_small_probe.py --cache-frames 18
python scripts/guard_lingbot_small_probe.py --cache-frames 70
```

The supervisor budgets 45GB of application memory plus 16GiB reserve, starts only when the local GPU is idle, and kills only its own child on a reserve breach. No extra cloud GPU job is used. Native fixed-camera replays are transport/quality diagnostics, not adaptive PlayWorld scores.

Both complete13-chunk runs finished with all native camera audits passing. Four initial chunks are bit-exact before the18-frame cache begins evicting. The warmed last-four means are12.40seconds for18frames versus17.52seconds for70frames; the larger context is slower. Total times391/359seconds reflect uncontrolled cold-load differences. No context-quality benefit is inferred without blinded review. See comparison.json and the two complete videos.

## Gemini diagnostic judgment

The frozen two-video comparison scored **1.2 for18/6 versus3.8 for70/6** using the released Gemini verifier via Vertex3.1Pro-preview. Both gates say partial completion and **no revisit**: neither ends back at the initial sports-center view. A higher consistency rubric score is not proof of world recovery or a player-driven benchmark improvement. This is one case/seed and one judge draw per video.

The [first/middle/final-frame comparison](context-contact.jpg) shows the floor expanding and changing perspective; the initial pink building is absent at the end in both arms. The larger cache preserves a more legible court in this sample, but the requested360-degree return still fails. Review both complete videos and [raw judgments](gemini-summary.json); camera travel and dynamic state need independent confirmation.

Rejudge with `.venv-lingbot/bin/python scripts/evaluate_lingbot_small_context.py`. The task, native fixed controls, seed, frame count, cache sizes and video hashes are validated before judging. These results stay separate from adaptive PlayWorld scores and from the14B experiments.

[Scoring audit](scoring-audit.md): the released gate admits partial trajectories. The3.8 is visual rubric credit on an incomplete journey, not demonstrated memory.
