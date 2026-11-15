"""Isolated Commit Forcing transfer to native LingBot four-latent chunks.

Only each attached instance's attention lookup is changed; native cache writes,
positions, denoising reads, backbone weights and the global module stay intact.
"""
import functools
import inspect
import types

import torch


class CommitForcing:
    def __init__(self, pipeline, *, enabled=False, chunk_latent_frames=4):
        self.model = pipeline.model
        self.enabled = enabled
        if chunk_latent_frames != 4:
            raise ValueError("Only native four-latent LingBot chunks are validated")
        self.chunk_frames = chunk_latent_frames
        self.clean = False
        self.bindings = []
        self.module_bindings = []
        self.handles = []
        self.trace = []
        self.closed = False
        self.active = False

    def _attention(self, original, layer, q, k, v, *args, **kwargs):
        read_k, read_v = k, v
        if self.clean:
            if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
                raise ValueError('Commit Forcing expects native batch/sequence/head/channel tensors')
            if q.shape[1] % self.chunk_frames or k.shape[1] < q.shape[1]:
                raise ValueError('Native cache must include the full current four-latent chunk')
            if args or any(kwargs.get(name) is not None for name in ('q_lens','k_lens')):
                raise ValueError('Variable-length native self-attention needs a separate audited policy')
            # Native inserts the current block before this attention call. Keep
            # its self-tokens AND one stored predecessor, at original positions.
            width = min(k.shape[1], 2 * q.shape[1])
            if width < k.shape[1]:
                read_k, read_v = k[:, -width:], v[:, -width:]
        self.trace.append({'layer': layer, 'clean': self.clean,
                           'query_tokens': q.shape[1], 'available_key_tokens': k.shape[1],
                           'read_key_tokens': read_k.shape[1]})
        return original(q, read_k, read_v, *args, **kwargs)

    def __enter__(self):
        if self.active or self.closed:
            raise RuntimeError('Commit Forcing context cannot be reused')
        if not self.enabled:
            return self
        try:
            for index, block in enumerate(self.model.blocks):
                layer = block.self_attn
                if hasattr(layer, 'memory_reader'):
                    raise ValueError('Learned-reader and Commit Forcing ablations must run separately')
                if 'forward' in layer.__dict__:
                    raise ValueError('Commit Forcing needs unwrapped native self-attention; run ablations separately')
                native = layer.forward
                if (not inspect.ismethod(native) or native.__func__.__module__ != 'wan.modules.model_fast'
                        or 'attention' not in native.__func__.__globals__):
                    raise ValueError('Only pinned native LingBot self-attention is supported')
                fn = native.__func__
                original = fn.__globals__['attention']
                if not inspect.isfunction(original):
                    raise ValueError('An attention-memory wrapper is already active; run isolated ablations')
                namespace = dict(fn.__globals__)
                if not any(values is fn.__globals__ for values, _ in self.module_bindings):
                    self.module_bindings.append((fn.__globals__, original))
                namespace['attention'] = functools.partial(self._attention, original, index)
                copied = types.FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__)
                functools.update_wrapper(copied, fn)
                copied.__kwdefaults__ = fn.__kwdefaults__
                self.bindings.append(layer)
                layer.forward = types.MethodType(copied, layer)
            self.handles.append(self.model.register_forward_pre_hook(self._before, with_kwargs=True))
            self.handles.append(self.model.register_forward_hook(self._after, with_kwargs=True, always_call=True))
            self.active = True
            return self
        except BaseException:
            self.close()
            raise

    def _before(self, module, positional, keywords):
        if (any(values['attention'] is not original for values, original in self.module_bindings)
                or any(hasattr(layer, 'memory_reader') for layer in self.bindings)):
            raise ValueError('Attention binding changed while Commit Forcing is active; run isolated ablations')
        timestep = keywords.get('t')
        if timestep is None and len(positional) > 1:
            timestep = positional[1]
        if timestep is None:
            raise ValueError('A complete clean-pass timestep is required')
        self.clean = bool(torch.all(timestep == 0))

    def _after(self, *_):
        self.clean = False

    def close(self):
        if self.closed:
            return
        for handle in self.handles:
            handle.remove()
        for layer in self.bindings:
            del layer.forward
        self.handles.clear()
        self.bindings.clear()
        self.module_bindings.clear()
        self.clean = False
        self.active = False
        self.closed = True

    def metadata(self):
        return {'enabled': self.enabled, 'closed': self.closed,
                'scope': 'Unvalidated LingBot transfer; not a paper reproduction or quality result',
                'committing_context': 'current chunk plus one stored predecessor; first chunk self-only',
                'denoising_context': 'unchanged native context',
                'chunk_latent_frames': self.chunk_frames,
                'maximum_clean_read_latent_frames': 2 * self.chunk_frames,
                'trace': self.trace}

    def __exit__(self, *_):
        self.close()
