"""Diagnostic mirror of native scalar cache indices; preserve all tensor writes."""


class HostCacheIndex:
    def __init__(self, tensor, initial=0):
        self.tensor = tensor
        self.value = int(initial)

    def item(self):
        return self.value

    def fill_(self, value):
        self.value = int(value)
        self.tensor.fill_(value)
        return self


def install(upstream):
    """Native pinned initializer creates these two indices at known zero."""
    original = upstream.WanI2VCausal._initialize_self_kv_cache

    def initialize(self, *args, **kwargs):
        caches = original(self, *args, **kwargs)
        for cache in caches:
            for name in ("global_end_index", "local_end_index"):
                cache[name] = HostCacheIndex(cache[name], initial=0)
        return caches

    upstream.WanI2VCausal._initialize_self_kv_cache = initialize
