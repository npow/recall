"""Memory v2, latents-only variant: recompute the K/V of retrieved frames instead of archiving them.

Keeps each chunk's clean latent, image condition and camera embedding on the CPU (a few MB per
chunk) and, when frames are retrieved, passes each contiguous run of them (at most one chunk)
through the model at t=0, at its native positions, with its own camera embedding and the current
text cache, attending to the live sink and to itself. Canonical keys and values are captured as
in the clean pass. Recomputed K/V differ from the originals because the original context is gone;
that is the risk this variant has to be measured against `--memory-v2 disk`.
"""
import time

import torch

from worldsync.lingbot_memv2 import MemoryV2


class _NoStore:
    bytes_written = 0
    write_seconds = 0.0
    read_seconds = 0.0
    frames_read = 0

    def read(self, index):
        return None  # filled by MemoryV2Recompute.prepare()

    def put(self, *args):
        raise RuntimeError("the recompute variant does not archive K/V")

    def close(self):
        pass


class MemoryV2Recompute(MemoryV2):
    def __init__(self, pipeline, **kwargs):
        kwargs.pop("store_root", None)
        super().__init__(pipeline, **kwargs)
        self.store.close()
        self.store = _NoStore()
        self.pipeline_model = pipeline.model
        self.layers = len(pipeline.model.blocks)
        self.clean_inputs = {}   # chunk -> (latent, image condition, camera embedding) on CPU
        self.clean_t = None
        self.recomputing = False
        self.recompute_frames = None
        self.recompute_sink = None
        self.recompute_seconds = 0.0
        self.recomputed_frames = 0

    def store_clean(self, chunk, kwargs):
        """Keep the clean latent and conditioning of a finished chunk. Call at its t=0 model call."""
        if chunk in self.clean_inputs:
            return
        self.clean_t = kwargs["t"].detach().clone()
        self.clean_inputs[chunk] = (kwargs["x"][0].detach().to("cpu", copy=True),
                                    kwargs["y"][0].detach().to("cpu", copy=True),
                                    kwargs["dit_cond_dict"]["c2ws_plucker_emb"][0].detach().to("cpu", copy=True))
        for offset in range(4):
            index = chunk * 4 + offset
            if index >= self.sink and not (index - self.sink) % self.stride:
                self.available.add(index)
                self.old_available.add(index)

    def prepare(self, kwargs):
        """Recompute K/V for newly selected frames. Call right after begin(), inside the model pre-hook."""
        missing = [i for i in self.selected if not isinstance(self.host.get(i), torch.Tensor)]
        if not missing:
            return
        started = time.monotonic()
        frame_tokens = int(kwargs["frame_seqlen"])
        device = kwargs["x"][0].device
        real = kwargs["kv_cache"]
        runs, run = [], [missing[0]]
        for index in missing[1:]:
            if index == run[-1] + 1 and index // 4 == run[0] // 4:
                run.append(index)
            else:
                runs.append(run)
                run = [index]
        runs.append(run)
        self.recomputing = True
        try:
            for run in runs:
                chunk, lo = run[0] // 4, run[0] % 4
                hi = lo + len(run)
                latent, condition, camera = self.clean_inputs[chunk]
                shape = list(real[0]["k"].shape)
                shape[1] = len(run) * frame_tokens
                scratch = [{"k": torch.zeros(shape, dtype=real[0]["k"].dtype, device=device),
                            "v": torch.zeros(shape, dtype=real[0]["v"].dtype, device=device),
                            "global_end_index": torch.tensor([run[0] * frame_tokens], dtype=torch.long, device=device),
                            "local_end_index": torch.tensor([0], dtype=torch.long, device=device)}
                           for _ in range(self.layers)]
                self.recompute_frames = run
                self.recompute_sink = real
                for index in run:
                    self.host[index] = torch.empty((self.layers, 2, frame_tokens, shape[2], shape[3]), dtype=torch.bfloat16)
                self.pipeline_model(x=[latent[:, lo:hi].to(device)], t=self.clean_t.to(device), context=kwargs["context"],
                                    seq_len=kwargs["seq_len"], y=[condition[:, lo:hi].to(device)],
                                    dit_cond_dict={"c2ws_plucker_emb": [camera[:, :, lo:hi].to(device)]},
                                    kv_cache=scratch, crossattn_cache=kwargs["crossattn_cache"],
                                    current_start=run[0] * frame_tokens, max_attention_size=kwargs["max_attention_size"],
                                    frame_seqlen=frame_tokens, cross_attn_first_call=False)
                del scratch
                self.recomputed_frames += len(run)
        finally:
            self.recomputing = False
            self.recompute_frames = None
            self.recompute_sink = None
        elapsed = time.monotonic() - started
        self.recompute_seconds += elapsed
        self.trace[-1]["recompute_seconds"] = elapsed
        self.trace[-1]["recomputed_frame_ids"] = missing

    def attend(self, query, key, value, **kwargs):
        if self.recomputing and self.active is not None:
            layer, start, frame_tokens, grid, freqs = self.active
            if self.raw_key is None:
                raise RuntimeError("canonical key capture missing during recompute")
            raw = self.raw_key.view_as(query)
            v_current = value[:, -query.shape[1]:]
            for offset, index in enumerate(self.recompute_frames):
                lo, hi = offset * frame_tokens, (offset + 1) * frame_tokens
                self.host[index][layer, 0] = raw[0, lo:hi].detach().to("cpu")
                self.host[index][layer, 1] = v_current[0, lo:hi].detach().to("cpu")
            sink_tokens = self.sink * frame_tokens
            return self.original_attention(query,
                torch.cat([self.recompute_sink[layer]["k"][:, :sink_tokens], key], dim=1),
                torch.cat([self.recompute_sink[layer]["v"][:, :sink_tokens], value], dim=1), **kwargs)
        if self.clean:
            # Latents are kept by store_clean(); there is nothing to archive per layer.
            self.clean = False
            try:
                return super().attend(query, key, value, **kwargs)
            finally:
                self.clean = True
        return super().attend(query, key, value, **kwargs)

    def metadata(self):
        data = super().metadata()
        data.update({"method": "memory v2 (recompute): clean latents kept, K/V of retrieved frames recomputed at t=0; diagnostic",
                     "kv_source": "recompute", "recompute_seconds": self.recompute_seconds,
                     "recomputed_latent_frames": self.recomputed_frames,
                     "clean_input_bytes": sum(sum(t.numel() * t.element_size() for t in item)
                                              for item in self.clean_inputs.values())})
        return data
