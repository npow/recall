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
