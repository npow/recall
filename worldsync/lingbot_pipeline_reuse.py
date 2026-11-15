"""Serial native-request pipeline reuse, with explicit state and config checks."""
import copy
import gc
import time


def parameter_versions(pipeline):
    return tuple((name, int(parameter._version)) for name, parameter in pipeline.model.named_parameters())


def hook_count(pipeline):
    return sum(len(module._forward_pre_hooks) + len(module._forward_hooks)
               for module in pipeline.model.modules())


class PipelineReuse:
    """One model/config per serial context; never share across concurrent calls."""
    def __init__(self, rollout):
        self.rollout = rollout
        self.original_backend = rollout.configure_backend
        self.pipeline = None
        self.constructor_calls = 0
        self.upstream = None

    def __enter__(self):
        self.rollout.configure_backend = self.configure
        return self

    def configure(self, report, save):
        upstream = self.original_backend(report, save)
        if self.upstream is None:
            self.upstream = upstream
            self.constructor = upstream.WanI2VCausal

        def construct(*args, **kwargs):
            signature = (tuple(repr(value) for value in args),
                         tuple(sorted((name, repr(value)) for name, value in kwargs.items())))
            started = time.monotonic()
            reused = self.pipeline is not None
            if reused:
                if signature != self.signature:
                    raise ValueError("Different pipeline settings require a separate batch worker")
                self.audit(require_encoder_cpu=True)
                self.pipeline.clear_text_cache()
                self.pipeline.vae.model.clear_cache()
                report["checkpoint_loading"] = copy.deepcopy(self.loading_info)
            else:
                # Drop cycles left by a previous failed request before loading.
                import torch
                gc.collect()
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
                self.pipeline = self.constructor(*args, **kwargs)
                self.signature = signature
                self.loading_info = copy.deepcopy(report["checkpoint_loading"])
                self.versions = parameter_versions(self.pipeline)
                self.initial_hooks = hook_count(self.pipeline)
                self.constructor_calls += 1
            report["persistent_worker"] = {"pipeline_reused": reused,
                "constructor_or_reuse_wall_s": time.monotonic() - started,
                "text_cache_entries_before_generation": len(self.pipeline._t5_cache),
                "encoder_device_before_generation": str(next(self.pipeline.text_encoder.model.parameters()).device)}
            save()
            return self.pipeline

        upstream.WanI2VCausal = construct
        return upstream

    def audit(self, require_encoder_cpu=True):
        if self.pipeline is None:
            return {"pipeline_loaded": False}
        state = {"pipeline_loaded": True, "remaining_model_hooks": hook_count(self.pipeline),
                 "dit_parameter_versions_unchanged": parameter_versions(self.pipeline) == self.versions,
                 "encoder_device_after_request": str(next(self.pipeline.text_encoder.model.parameters()).device),
                 "text_cache_entries_after_request": len(self.pipeline._t5_cache)}
        if (state["remaining_model_hooks"] != self.initial_hooks
                or not state["dit_parameter_versions_unchanged"]
                or (require_encoder_cpu and state["encoder_device_after_request"] != "cpu")):
            raise ValueError(f"Unsafe persistent pipeline state: {state}")
        return state

    def discard(self):
        if self.pipeline is not None:
            self.pipeline.clear_text_cache()
            self.pipeline.vae.model.clear_cache()
        self.pipeline = None
        gc.collect()

    def __exit__(self, *_):
        self.rollout.configure_backend = self.original_backend
        if self.upstream is not None:
            self.upstream.WanI2VCausal = self.constructor
        self.discard()
