# Native camera round trip and first retrieval comparison

**Result: reject this retrieval variant. Neither video realized the commanded
return, and endpoint similarity was slightly worse with retrieval.** This is
one development scene/seed, not a PlayWorld score or a memory-method success.

GC007 image, seed 42, official LingBot-World2 14B causal-fast, one H200 per run.
Fixed path: WAIT; forward 4 s; right yaw 180 degrees over 3 s; forward 4 s;
right yaw 180 degrees over 3 s; WAIT. Both end at the commanded origin and
heading. Commanded closure does not prove realized visual closure.

## Camera integration correction

The first native-control audit stopped at the first turn: the online hook was
computing ray matrices inside BF16 autocast, whereas native inference builds
camera rays in FP32 before entering autocast. Neutral-only tests had missed
this. Protocol v4 disables autocast for ray computation. On the completed
16-chunk path, the online embeddings match native within approximately 1e-16.
Both videos here use the native pose-file loader, not substituted online rays.
The earlier v3 episode is preserved but is not the corrected-native baseline.

## Intervention and controls

The diagnostic archives 16 early non-sink latent frames on CPU, storing
canonical pre-RoPE keys and native values. When the commanded view overlaps,
four historical frames replace four older recent frames; retrieved keys receive
virtual temporal positions before attention. The attended budget stays at 18
latent-frame equivalents: sink 6, retrieved 4, recent/current 8. This is inspired
by WorldKV/WorldTrace, not a reproduction of either paper or DeCoPrune.

Both runs use identical camera poses, source/checkpoint, sampling, seed,
resolution and caption. All first **14 chunks have identical raw pixel hashes**.
Retrieval activates only at chunks 14 and 15, reading historical frames 6–9.
The archive occupies 19.77 GB of CPU tensors, and peak CUDA allocation rises
from 80.39 to 85.33 GB. The archive is bounded but covers only early history;
it cannot support arbitrary later locations or reliable dynamic-state evolution.

| Endpoint diagnostic | Stock | Retrieval |
| --- | ---: | ---: |
| DINOv2-small CLS cosine, higher better | 0.6077 | 0.5999 |
| PSNR dB, higher better | 9.599 | 9.072 |
| Actual round trip, blinded Claude review | No | No |
| Starting viewpoint recovered | No | No |

These metrics compare the last frame of the initial hold with the last frame
of the final hold. They cannot be interpreted as pure memory fidelity because
neither video executes the intended return. The blinded judge preferred neither:
both largely hold the initial viewpoint, then drift into an arcade and plaza
instead of turning around and retracing their route. The judge's detailed
observations are retained as model assessments, alongside videos for inspection.
Our contact-sheet review also shows that the final viewpoint does not match
the original mall facade.

Next: isolate translation and rotation with native controls. Do not proceed to
more memory tuning on the assumption that forgetting is the only bottleneck.

## Completed control-isolation results

Both predeclared isolation paths completed with native camera conditioning and
passed the per-chunk online/native input audit. The complete-video Claude review
and our contact-sheet inspection find:

* **Rotation-only (7.8125 s):** a visible pan and direction reversal occur, but
  the returning mall facade, roof, signs and layout morph or duplicate. Exact
  180-degree execution is not established, and there is some apparent drift.
  This exposes geometric/identity inconsistency as well as imperfect control.
* **Translation-only (9.8125 s):** the forward phase produces little visible
  approach; a late retreat-like change introduces foreground content. The
  requested forward/backward round trip is not visually demonstrated. Similar
  frames during the frozen phase must not be credited as memory success.

These are one-scene/one-seed calibration observations with different path
durations, not a statistical comparison between control axes. The next useful
control tests are small-amplitude yaw loops and translation-only sequences with
near-field landmarks and measurable parallax. No further tuning has been
declared successful, and the baseline/control gate remains closed.

* [Rotation-only video](turn.mp4), [contact sheet](turn-contact.jpg), [run](turn-run.json)
* [Translation-only video](translate.mp4), [contact sheet](translate-contact.jpg), [run](translate-run.json)
* [Full independent control review, prompt and sampling protocol](control-isolation-review.json)

* [Stock video](stock.mp4) and [contact sheet](stock-contact.jpg)
* [Retrieval video](retrieval.mp4) and [contact sheet](retrieval-contact.jpg)
* [Comparison, metrics, complete blinded assessment](comparison.json)
* [Stock controls and hashes](stock-run.json)
* [Retrieval controls, history selection and hashes](retrieval-run.json)

Run commands:

```bash
modal run --profile nissan-pow --detach scripts/modal_lingbot_roundtrip.py
modal run --profile nissan-pow --detach scripts/modal_lingbot_roundtrip.py --retrieval
modal run --profile nissan-pow --detach scripts/modal_lingbot_roundtrip.py --path-name turn_return
modal run --profile nissan-pow --detach scripts/modal_lingbot_roundtrip.py --path-name translate_return
```
