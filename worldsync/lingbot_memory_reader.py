"""Experimental learned residual read from retrieved KV; no default behavior change.

This is an explicit-memory reader, not Titans or an NTM reproduction. Projection
weights are shared across heads within each layer; layers have separate readers.
Inputs use the existing virtual RoPE convention so training and retrieval agree.
"""
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class MemoryReader(nn.Module):
    def __init__(self, head_dim, rank=32, device=None):
        super().__init__()
        if not 0 < rank <= head_dim:
            raise ValueError("reader rank must be between 1 and head dimension")
        self.rank = rank
        self.q = nn.Linear(head_dim, rank, bias=False, device=device, dtype=torch.float32)
        self.k = nn.Linear(head_dim, rank, bias=False, device=device, dtype=torch.float32)
        self.v = nn.Linear(head_dim, rank, bias=False, device=device, dtype=torch.float32)
        self.out = nn.Linear(rank, head_dim, bias=False, device=device, dtype=torch.float32)
        nn.init.zeros_(self.out.weight)

    def forward(self, query, key, value):
        if key.shape != value.shape or query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
            raise ValueError("memory and queries must share batch, heads and head dimension")
        if not key.shape[1]:
            return torch.zeros_like(query)
        # Honor the enclosing bf16 autocast in training/inference; master weights
        # remain fp32. Fused SDPA avoids materializing a token-by-token matrix.
        q, k, v = self.q(query.float()), self.k(key.float()), self.v(value.float())
        read = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.out(read.transpose(1, 2)).to(query.dtype)


def attach(model, rank=32):
    if any(hasattr(block.self_attn, "memory_reader") for block in model.blocks):
        raise ValueError("memory reader already attached")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = []
    for block in model.blocks:
        attention = block.self_attn
        reader = MemoryReader(attention.head_dim, rank, attention.q.weight.device)
        attention.memory_reader = reader
        parameters.extend(reader.parameters())
    return parameters


def save(model, path, extra=None):
    readers = [block.self_attn.memory_reader for block in model.blocks]
    torch.save({"format": "worldsync-memory-reader-v1", "rank": readers[0].rank,
                "layers": len(readers), "head_dim": readers[0].q.in_features,
                "state": {str(i): reader.state_dict() for i, reader in enumerate(readers)},
                "extra": extra or {}}, path)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(model, path):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if (blob["format"] != "worldsync-memory-reader-v1" or blob["layers"] != len(model.blocks)
            or any(block.self_attn.head_dim != blob["head_dim"] for block in model.blocks)
            or set(blob["state"]) != {str(i) for i in range(len(model.blocks))}):
        raise ValueError("reader checkpoint does not match this model")
    if not any(hasattr(block.self_attn, "memory_reader") for block in model.blocks):
        attach(model, blob["rank"])
    for i, block in enumerate(model.blocks):
        block.self_attn.memory_reader.load_state_dict(blob["state"][str(i)], strict=True)
    return {"rank": blob["rank"], "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
