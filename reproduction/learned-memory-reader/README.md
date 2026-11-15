# Learned memory reader: feasibility, not a benchmark result

The opt-in reader adds 655,360 parameters to frozen LingBot 14B. It reads the same four historical KV views used by our existing retrieval method through a separate low-rank attention branch. Its zero output projection initially preserves **the existing retrieval baseline**, not the stock 18-frame cache once retrieval replaces old slots. [Architecture and rationale](../../docs/differentiable-memory.md).

## Completed synthetic plumbing probe

One 14B update on a synthetic pan made from GC001 completed on Thor. This checkpoint is **plumbing-only**, trained on a benchmark image, and must not be promoted or used as a held-out benchmark candidate.

| Check | Evidence |
|---|---:|
| Trainable parameters |655,360|
| First-step gradient norm |0.0408814|
| Update time |48.37s|
| CUDA allocated peak |63.73GB|
| Minimum physical RAM available |30.69GB|
| Teacher/native relative mean forward error |0.674%|
| Checkpoint reload |Exact fixed-noise loss match|

The teacher/native gap is the existing BF16/recomputed-context harness gap, not a claim of full-model bit equivalence. Base weights are frozen, the optimizer receives only reader parameters, and the base-gradient guard passed. The probe did not check every base weight byte.

At the fixed evaluation noise/schedule point, correct-memory flow loss changed0.574399 → 0.574548; wrong-memory loss 0.582969 → 0.583336. Absent-memory loss stayed exactly 0.582081. Zeroing the trained reader restores the initial correct-memory loss; reloading restores the trained value exactly. This demonstrates a working update and isolated branch, **not improved quality**. Training used a different schedule point from the one-point evaluation.

[Raw training report](synthetic-gradient-probe/train.json), supervisor record, full log, physical memory samples and the small adapter checkpoint are retained. Reproduce its source at commit df7562c; the report pins each training-source SHA256. The full probe took 560s including model load, encoding, parity checks, evaluations and checkpoint round-trip.

## Frozen real-data experiment

[The plan](initial-real-data-plan.json) selects eight real revisit examples by a fixed hash from eligible content-checked pairs, keeping source videos disjoint: six training and two held out. The 24 input files and selected manifest are hashed. Matched-size wrong-memory controls are fixed before fitting; they are earlier nonoverlapping chunks, not guaranteed semantically dissimilar ones.

The planned fit starts from fresh pinned14B weights, uses rank 32, 60 updates, learning rate 1e-4 and seed 0. It never loads the synthetic probe adapter. Endpoint-schedule/single-noise evaluation is an explicitly limited feasibility surrogate, not a PlayWorld score. All held-out outcomes must be reported; no checkpoint selection using their losses is allowed. T5 is discarded after encoding to free physical memory.

The preflight validates the dataset and frozen split without a GPU launch:

```sh
.venv-lingbot/bin/python scripts/guard_lingbot_reader_smoke.py --real --preflight-only
```

Run `--real` without `--preflight-only` only when the local GPU is free. The supervisor enforces the 16 GiB physical reserve and stops only its own child on a breach. Passing this experiment still requires player-driven quality confirmation and a full declared evaluation before any claim against 2.12.

## Execution

The local fit supervisor refused to start because another collaborator owned Thor; [that refusal is retained](local-fit-not-launched.json). The same frozen fit is queued as oneH200 job after a CPU preflight verified all24 input hashes and the exact6/2 source split. [Execution record](execution.json). No local fit was launched, and the synthetic plumbing weights are not used.

The [real-data contact sheet](real-data-contact.jpg) shows the selected memory/revisit frames. Readable overlapping views do not establish that recent context lacks the target information; the correct/wrong/absent-memory controls remain necessary, and teacher-forced loss is not a generated-world result.
