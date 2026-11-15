# Persistent native research batches

Use the batch launcher to reuse a pretrained LingBot pipeline across serial fixed-path experiments. Multiple initialization groups still run in parallel, with at most three H100 workers. This is a research batch interface, not an HTTP service or an adaptive-player API.

```bash
.venv-modal-new/bin/modal run --profile nissan-pow scripts/modal_lingbot_batch.py \
  --plan reproduction/lingbot-batch-validation-plan.json \
  --output-root reproduction/lingbot-batch-validation \
  --jobs-per-worker 3
```

A plan contains `jobs`, each with a unique simple `id` and an `argv` list using the existing rollout CLI. Omit `--output`; the worker manages isolated result folders. Jobs currently require `--split gc`, `--native-fixed-controls`, `--fixed-actions ...`, and `--offload-text-encoder`. The image contains the complete GC input set. Inference stays on the established Torch2.6/CUDA12.4 SDPA runtime with pinned model weights and source.

```json
{
  "scope": "my declared memory ablation",
  "jobs": [
    {
      "id": "GC007-seed42-memory4",
      "argv": ["--split", "gc", "--task-id", "GC007", "--seed", "42",
               "--max-chunks", "2", "--native-fixed-controls",
               "--fixed-actions", "WAIT", "WAIT", "--pose-retrieval",
               "--offload-text-encoder"]
    }
  ]
}
```

That short example only exercises transport; longer scored paths are needed to test memory. Copy complete declared paths from experiment plans for actual comparisons.

Grouping uses exact source/checkpoint/backend/cache/sink settings, then splits each group into batches of1–4 requests. The factory additionally checks the actual constructor signature. Changing image, camera path, seed or memory policy can reuse the same pipeline; changing cache initialization uses a separate worker. No two requests execute concurrently inside a pipeline.

The native generator creates fresh self/cross-attention KV caches and seeded noise for each request. The reuse context clears prompt and VAE feature caches before reuse, checks model parameter version counters and residual hooks, and preserves the fresh-request encoder placement. Existing rollout cleanup removes memory hooks. A failed request is retained and its pipeline discarded before the next job; later jobs still run and all failures remain in the final denominator.

Optional `reference_manifest` paths load complete fresh-run manifests before GPU allocation. These validation jobs require matched generation settings and every raw chunk hash to match. A missing or differing reference cannot silently pass. Without a reference, complete native generation and state checks still run, but no pixel-parity claim is made.

Completed per-job folders are committed to the Modal results volume while subsequent requests run. The local result folder receives each `run.json`, group-level failures/results and a summary. Reusing an output root is rejected to preserve prior evidence. Each remote group records its volume directory, declared job ids, constructor count, request times and checks.

The earlier [single-image reuse probe](../reproduction/lingbot-persistent-worker/README.md) passed exact repeated-seed/policy isolation and achieved2.27× memory-trial throughput by removing repeated checkpoint loading. The [mixed-image, moving-camera validation](../reproduction/lingbot-batch-validation/README.md) now independently passes all42 raw chunk hashes for GC010, GC033 and GC040 with one constructor. All three use832×464 output; different-aspect-ratio reuse remains unverified.
