The complete local baseline and lossless four-view episode both finished on NVIDIA Thor, Torch 2.15/cu132. Each generated all 13 chunks (205 frames) for GC027, seed 42, WAIT followed by 12 RIGHT commands at 30 degrees/second, cache 18/sink 6, unchanged BF16 parameters/native KV. Both discard the 11.36GB T5 model after encoding and enforce a 16GiB physical-memory reserve.

All twelve chunks before the first real retrieval intervention have exactly identical raw pixel hashes. Retrieval activates at chunk 12, where the outputs differ. This verifies transport isolation and local feasibility; it does not demonstrate improved quality or reproduce the H200/Torch 2.6 episode numerically.

| Measurement | No archive | Lossless file-backed four views |
|---|---:|---:|
| Total harness wall time |834.10s|986.43s|
| Native CUDA allocation peak |69.02GB|72.53GB|
| Sampled CUDA reserved peak |71.08GB|74.11GB|
| Peak process RSS, including loading |61.84GB|61.47GB|
| Minimum physical memory available |26.18GB|34.37GB|
| Completed chunks / native audits |13/13|13/13|

These are separate runs with changing background CPU activity, so available-memory and startup differences are not controlled performance comparisons. During archive operation candidate RSS stayed around 2.9GB, while 40 flushed/evicted mappings held 19.7657GB of original BF16 features. Filesystem caches remain physical, reclaimable memory; dropping mappings does not make them disappear. The final 4.9414GB retrieved GPU buffer was released before VAE decode. Synchronous flushes consumed 99.20s, so this implementation enables local experiments rather than accelerating them. A deferred-flush variant requires separate residency validation.

The adapter file hashes differ because streaming-player changes landed between the two launches. Those changes are inactive for fixed native controls. The upstream source, checkpoint, image/caption, sampler, seed, camera command trace, shapes and attention backend match; all pre-intervention pixels also match exactly. The reporter records this hash difference rather than treating the pair as a generic exact-code benchmark.

Reproduction commands:

```sh
python scripts/guard_lingbot_thor_smoke.py --episode baseline
python scripts/guard_lingbot_thor_smoke.py --episode file4
python scripts/report_lingbot_thor_pair.py BASELINE_OUTPUT FILE4_OUTPUT
```

The supervisor checks an idle local GPU, fresh MemAvailable, at least 25GB free disk for the file candidate, and an 87.61GB application estimate plus 16GiB reserve. It terminates only its own child if the physical reserve is breached. The file bank is an isolated prototype; normal rollout defaults are unchanged. Full videos, manifests, logs and both memory/IO monitors are retained here. The 19.8GB feature files remain in ignored local outputs and are not committed.

The raw baseline result.scope retains the old two-chunk smoke label from the harness version loaded at launch. Its episode field is baseline, and the actual run records max_chunks=13, thirteen chunk hashes and205 generated frames. The raw artifact is preserved unchanged; the measured episode was complete.
