"""Minimal LoRA for LingBot-World2's attention projections (no peft dependency).

Wraps the q/k/v/o linears of every block's self-attention and cross-attention.
B is zero-initialised, so a freshly attached adapter is exactly the stock model.
Only the adapter tensors are saved/loaded, as one torch file with metadata.
"""
import hashlib
from pathlib import Path

import torch
import torch.nn as nn

TARGETS = ("q", "k", "v", "o")
ATTENTION_MODULES = ("self_attn", "cross_attn")


class LoRALinear(nn.Module):
    """y = W x + b + (alpha / rank) * B(A(x)); W frozen, A/B trainable in fp32."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank, self.alpha = rank, alpha
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x):
        out = self.base(x)
        delta = torch.nn.functional.linear(torch.nn.functional.linear(x.to(self.lora_a.dtype), self.lora_a), self.lora_b)
        return out + (delta * self.scale).to(out.dtype)  # keep the stock output dtype (bf16 under autocast)


def _iter_targets(model):
    for index, block in enumerate(model.blocks):
        for module_name in ATTENTION_MODULES:
            module = getattr(block, module_name)
            for target in TARGETS:
                yield f"blocks.{index}.{module_name}.{target}", module, target


def apply_lora(model, rank=32, alpha=32.0):
    """Freeze the model and attach adapters. Returns the list of trainable parameters."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for _, module, target in _iter_targets(model):
        current = getattr(module, target)
        if isinstance(current, LoRALinear):
            raise ValueError("LoRA already attached")
        setattr(module, target, LoRALinear(current, rank, alpha))
    return trainable_parameters(model)


def trainable_parameters(model):
    return [p for name, p in model.named_parameters() if name.endswith(("lora_a", "lora_b"))]


def lora_state_dict(model):
    return {f"{name}.{part}": getattr(getattr(module, target), part).detach().cpu()
            for name, module, target in _iter_targets(model) for part in ("lora_a", "lora_b")}


def save_lora(model, path, extra=None):
    state = lora_state_dict(model)
    first = next(iter(state.values()))
    rank = first.shape[0]
    layer = getattr(next(iter(_iter_targets(model)))[1], "q")
    torch.save({"lora": state, "rank": rank, "alpha": layer.alpha, "targets": TARGETS,
                "modules": ATTENTION_MODULES, "extra": extra or {}}, str(path))
    return file_sha256(path)


def load_lora(model, path):
    """Attach adapters sized from the file and load them. Returns metadata."""
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    if tuple(blob["targets"]) != TARGETS or tuple(blob["modules"]) != ATTENTION_MODULES:
        raise ValueError("LoRA file targets differ from this build")
    if not any(isinstance(getattr(module, target), LoRALinear) for _, module, target in _iter_targets(model)):
        apply_lora(model, rank=blob["rank"], alpha=blob["alpha"])
    expected = {f"{name}.{part}" for name, _, _ in _iter_targets(model) for part in ("lora_a", "lora_b")}
    if set(blob["lora"]) != expected:
        raise ValueError("LoRA file does not match this model's layers")
    for name, module, target in _iter_targets(model):
        layer = getattr(module, target)
        layer.lora_a.data.copy_(blob["lora"][f"{name}.lora_a"])
        layer.lora_b.data.copy_(blob["lora"][f"{name}.lora_b"])
    return {"rank": blob["rank"], "alpha": blob["alpha"], "sha256": file_sha256(path), "extra": blob.get("extra", {})}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(1 << 22):
            digest.update(block)
    return digest.hexdigest()
