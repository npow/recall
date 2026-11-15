"""Warm LingBot runtime: load the 14B pipeline once, run many rollouts against it.

`worldsync.lingbot_rollout.main(argv, runtime=WarmRuntime())` behaves exactly like the
one-shot CLI except that the (slow) pipeline construction happens only on the first job.
All per-run state is either created fresh by upstream `generate()` (KV caches, cross-attention
cache, noise from a per-call seeded generator, scheduler timesteps) or by `main()` (hooks,
memory banks, camera plan, VAE streaming cache) and removed in its `finally`. What a warm
process must additionally manage is handled here:

* cache geometry: `local_attn_size` / `sink_size` are plain int attributes on the pipeline and
  every self-attention module, so a job with a different window re-points them;
* the T5 prompt-embedding cache is keyed by prompt and returns the same tensors, so it is kept;
* the text encoder may have been offloaded to CPU by a previous job; upstream `generate()` moves
  it back to the GPU whenever it has to encode a new prompt;
* between jobs: drop references, `gc.collect()`, `torch.cuda.empty_cache()`.
"""
import gc
import time


def set_cache_geometry(pipeline, local_attn_size, sink_size):
    """Point the already-built model at a different rolling-window geometry."""
    pipeline.local_attn_size = local_attn_size
    pipeline.sink_size = sink_size
    for module in pipeline.model.modules():
        if hasattr(module, "local_attn_size"):
            module.local_attn_size = local_attn_size
        if hasattr(module, "sink_size"):
            module.sink_size = sink_size


def detach_lora(model):
    """Undo `lingbot_lora.apply_lora`: put the original linears back. Returns how many were removed."""
    removed = 0
    for block in model.blocks:
        for module_name in ("self_attn", "cross_attn"):
            module = getattr(block, module_name)
            for target in ("q", "k", "v", "o"):
                layer = getattr(module, target)
                if type(layer).__name__ == "LoRALinear":
                    setattr(module, target, layer.base)
                    removed += 1
    return removed


def cache_geometry(pipeline):
    """Return the set of (local_attn_size, sink_size) values currently held by the model."""
    pairs = {(pipeline.local_attn_size, pipeline.sink_size)}
    for module in pipeline.model.modules():
        if hasattr(module, "local_attn_size") and hasattr(module, "sink_size"):
            pairs.add((module.local_attn_size, module.sink_size))
    return pairs


class WarmRuntime:
    def __init__(self):
        self.pipeline = None
        self.checkpoint_loading = None
        self.load_seconds = None
        self.jobs_run = 0

    def pipeline_for(self, upstream, configs, pack, knobs, report):
        from worldsync.lingbot_rollout import build_pipeline
        if self.pipeline is None:
            started = time.monotonic()
            self.pipeline = build_pipeline(upstream, configs, pack, knobs)
            self.load_seconds = time.monotonic() - started
            self.checkpoint_loading = report.get("checkpoint_loading")
            report["warm_runtime"] = {"loaded_this_job": True, "load_seconds": self.load_seconds}
        else:
            if self.checkpoint_loading is not None:
                report["checkpoint_loading"] = self.checkpoint_loading
            report["warm_runtime"] = {"loaded_this_job": False, "load_seconds": 0.0}
        set_cache_geometry(self.pipeline, knobs["local_attn_size"], knobs["sink_size"])
        report["warm_runtime"]["job_index"] = self.jobs_run
        self.jobs_run += 1
        return self.pipeline

    def fingerprint(self):
        """Cheap integrity check that no job left the shared weights modified."""
        import torch
        if self.pipeline is None:
            return None
        blocks = self.pipeline.model.blocks
        picks = [blocks[0].self_attn.q.weight, blocks[0].self_attn.k.weight, blocks[len(blocks) // 2].self_attn.v.weight,
                 blocks[-1].self_attn.o.weight]
        with torch.no_grad():
            return [float(p.double().sum()) for p in picks]

    def between_jobs(self):
        """Release per-job garbage so every job starts from the same allocator-level state."""
        import torch
        if self.pipeline is not None:
            detach_lora(self.pipeline.model)  # a job's adapter must not leak into the next job
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
