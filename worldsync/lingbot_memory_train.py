"""Teacher-forced memory-keyframe training forward for LingBot-World2 14B causal-fast.

Inference attends a window of 18 latent frames: sink(6) + retrieved memory(4) +
previous chunk(4) + current chunk(4). This module reproduces that exact window
without the stateful KV cache so gradients can flow:

  * context groups (sink, memory, previous chunk) are clean (t=0) and run under
    no_grad layer-by-layer, each group causally attending to earlier groups,
    producing per-layer K/V at explicit RoPE frame positions (memory sits at the
    virtual slot [r-8, r-4), exactly where worldsync.lingbot_retrieval puts it);
  * the target chunk is noised at a timestep from the 4-step distilled sampler
    schedule and attends to context K/V plus itself; loss is flow matching on
    that chunk only. Block math mirrors CausalWanAttentionBlock.forward.

Known train/inference gaps (documented in reproduction/train-lora.md): context
K/V are recomputed here rather than archived from the original rollout; the
sink is a single group (the stock cache computes frames 4-5 inside chunk 1);
camera translations are normalised over the clip, not the rollout-so-far.
"""
import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

SINK, CHUNK = 6, 4
SCHEDULE_INDEX = (0, 250, 500, 750)  # timesteps_index of the distilled sampler


def rope_at(x, positions, h, w, freqs):
    """Apply 3D RoPE with explicit per-frame temporal positions. x: [1, f*h*w, n, d]."""
    n, c = x.size(2), x.size(3) // 2
    parts = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f = len(positions)
    seq = f * h * w
    index = torch.as_tensor(list(positions), device=freqs.device, dtype=torch.long)
    x_c = torch.view_as_complex(x[0, :seq].to(torch.float64).reshape(seq, n, -1, 2))
    table = torch.cat([
        parts[0][index].view(f, 1, 1, -1).expand(f, h, w, -1),
        parts[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        parts[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(seq, 1, -1)
    return torch.view_as_real(x_c * table).flatten(2).unsqueeze(0).type_as(x)


def time_embed(model, timestep):
    from wan.modules.model import sinusoidal_embedding_1d
    device = model.patch_embedding.weight.device
    with torch.amp.autocast("cuda", dtype=torch.float32, cache_enabled=False):
        t = torch.as_tensor([float(timestep)], device=device)
        e = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, t).unflatten(0, (1, 1)).float())
        e0 = model.time_projection(e).unflatten(2, (6, model.dim))
    return e, e0  # seq dim 1: broadcast over tokens, identical math to per-token t


def camera_hidden(model, plucker):
    from einops import rearrange
    emb = rearrange(plucker, "1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)", c1=1, c2=2, c3=2)
    emb = model.patch_embedding_wancamctrl(emb)
    return emb + model.c2ws_hidden_states_layer2(F.silu(model.c2ws_hidden_states_layer1(emb)))


class Group:
    """Tokens of a set of latent frames at explicit RoPE positions and timestep."""

    def __init__(self, model, latent, y, plucker, positions, timestep):
        tokens = model.patch_embedding(torch.cat([latent, y], dim=0).unsqueeze(0))
        self.frames, self.h, self.w = tokens.shape[2], tokens.shape[3], tokens.shape[4]
        self.x = tokens.flatten(2).transpose(1, 2)
        self.cam = camera_hidden(model, plucker)
        self.positions = list(positions)
        self.e, self.e0 = time_embed(model, timestep)
        assert len(self.positions) == self.frames


def layer_forward(block, x, group, freqs, text, context_kv=None):
    """One CausalWanAttentionBlock on `group`'s tokens; returns (x, k, v)."""
    with torch.amp.autocast("cuda", dtype=torch.float32, cache_enabled=False):
        e = (block.modulation.unsqueeze(0) + group.e0).chunk(6, dim=2)
    attn = block.self_attn
    b, s, n, d = 1, x.shape[1], attn.num_heads, attn.head_dim
    y = block.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
    q = attn.norm_q(attn.q(y)).view(b, s, n, d)
    k = attn.norm_k(attn.k(y)).view(b, s, n, d)
    v = attn.v(y).view(b, s, n, d)
    q = rope_at(q, group.positions, group.h, group.w, freqs).type_as(v)
    k = rope_at(k, group.positions, group.h, group.w, freqs).type_as(v)
    all_k, all_v = (k, v) if context_kv is None else (torch.cat([context_kv[0], k], 1), torch.cat([context_kv[1], v], 1))
    out = F.scaled_dot_product_attention(q.transpose(1, 2), all_k.transpose(1, 2), all_v.transpose(1, 2)).transpose(1, 2)
    out = attn.o(out.flatten(2))
    with torch.amp.autocast("cuda", dtype=torch.float32, cache_enabled=False):
        x = x + out * e[2].squeeze(2)
    hidden = block.cam_injector_layer2(F.silu(block.cam_injector_layer1(group.cam))) + group.cam
    x = (1.0 + block.cam_scale_layer(hidden)) * x + block.cam_shift_layer(hidden)
    x = x + block.cross_attn(block.norm3(x), text, None)
    y = block.ffn(block.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
    with torch.amp.autocast("cuda", dtype=torch.float32, cache_enabled=False):
        x = x + y * e[5].squeeze(2)
    return x, k, v


def device_freqs(model):
    device = model.patch_embedding.weight.device
    if model.freqs.device != device:
        model.freqs = model.freqs.to(device)  # same lazy move as WanModelFast.forward
    return model.freqs


def text_context(model, embedding):
    pad = embedding.new_zeros(model.text_len - embedding.size(0), embedding.size(1))
    return model.text_embedding(torch.cat([embedding, pad]).unsqueeze(0))


@torch.no_grad()
def context_kv(model, groups, text):
    """Per-layer (K, V) of the clean context groups, each attending to earlier groups."""
    hidden = [g.x for g in groups]
    per_layer = []
    for block in model.blocks:
        ks, vs = [], []
        for index, group in enumerate(groups):
            previous = (torch.cat(ks, 1), torch.cat(vs, 1)) if ks else None
            hidden[index], k, v = layer_forward(block, hidden[index], group, device_freqs(model), text, previous)
            ks.append(k)
            vs.append(v)
        per_layer.append((torch.cat(ks, 1), torch.cat(vs, 1)))
    return per_layer


def target_flow(model, target, per_layer, text, grad_checkpoint=True):
    """Predicted flow [16, F, H, W] (noise - x0) for the noisy target group."""
    x = target.x
    if torch.is_grad_enabled():
        x = x.detach().requires_grad_(True)  # a grad-carrying input keeps every checkpoint variant's graph connected
    for block, kv in zip(model.blocks, per_layer):
        if grad_checkpoint and torch.is_grad_enabled():
            x = checkpoint(lambda hidden, block=block, kv=kv: layer_forward(block, hidden, target, device_freqs(model), text, kv)[0],
                           x, use_reentrant=False)
        else:
            x = layer_forward(block, x, target, device_freqs(model), text, kv)[0]
    x = model.head(x, target.e)
    grid = torch.tensor([[target.frames, target.h, target.w]], dtype=torch.long)
    return model.unpatchify([x[0]], grid)[0].float()


class Clip:
    """Encoded training clip: clean latents, VAE conditioning, camera rays, text."""

    def __init__(self, latent, y, plucker, text, revisit_chunk, memory_start, wrong_start=None, name=""):
        self.latent, self.y, self.plucker, self.text = latent, y, plucker, text
        self.r, self.k, self.wrong = revisit_chunk, memory_start, wrong_start
        self.name = name
        if not (SINK + 2 <= memory_start and memory_start + CHUNK <= revisit_chunk - CHUNK):
            raise ValueError("memory chunk must lie after the sink and before the previous chunk")
        if revisit_chunk - 2 * CHUNK < SINK:
            raise ValueError("revisit chunk too early: virtual memory slot would overlap the sink")
        if revisit_chunk + CHUNK > latent.shape[1]:
            raise ValueError("clip shorter than the revisit chunk")


def _slice(clip, frames):
    index = torch.as_tensor(frames, device=clip.latent.device)
    return clip.latent[:, index], clip.y[:, index], clip.plucker[:, :, index]


def make_groups(model, clip, mode):
    """Context groups for the 18-frame inference window. mode: correct | none | wrong."""
    r = clip.r
    specs = [("sink", list(range(SINK)), list(range(SINK)))]
    if mode == "correct":
        specs.append(("memory", list(range(clip.k, clip.k + CHUNK)), list(range(r - 2 * CHUNK, r - CHUNK))))
    elif mode == "wrong":
        if clip.wrong is None:
            raise ValueError("clip has no wrong-memory chunk")
        specs.append(("memory", list(range(clip.wrong, clip.wrong + CHUNK)), list(range(r - 2 * CHUNK, r - CHUNK))))
    elif mode != "none":
        raise ValueError(mode)
    specs.append(("tail", list(range(r - CHUNK, r)), list(range(r - CHUNK, r))))
    return [Group(model, *_slice(clip, frames), positions, 0.0) for _, frames, positions in specs]


def schedule(pipeline):
    pipeline.scheduler.set_timesteps(pipeline.num_train_timesteps, shift=10)
    return pipeline.scheduler.timesteps[list(SCHEDULE_INDEX)]


def memory_loss(pipeline, clip, mode="correct", step_index=0, generator=None, loss_space="velocity",
                grad_checkpoint=True, cached=None):
    """Flow-matching loss on the revisit chunk at distilled-schedule step `step_index`."""
    model = pipeline.model
    timesteps = schedule(pipeline)
    t = timesteps[step_index]
    with torch.amp.autocast("cuda", dtype=pipeline.param_dtype, cache_enabled=False):  # checkpoint recompute must see the same casts
        text = text_context(model, clip.text)
        per_layer = cached if cached is not None else context_kv(model, make_groups(model, clip, mode), text)
        frames = list(range(clip.r, clip.r + CHUNK))
        x0, y, plucker = _slice(clip, frames)
        noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=torch.float32)
        noisy = pipeline.scheduler.add_noise(x0.float(), noise, t)
        target = Group(model, noisy, y, plucker, frames, t)
        flow = target_flow(model, target, per_layer, text, grad_checkpoint)
    if loss_space == "x0":
        estimate = pipeline._convert_flow_pred_to_x0(flow, noisy, t, pipeline.scheduler)
        loss = F.mse_loss(estimate.float(), x0.float())
    else:
        loss = F.mse_loss(flow, (noise - x0.float()))
    if torch.is_grad_enabled() and not loss.requires_grad:
        trainable = sum(1 for n, p in model.named_parameters() if n.endswith(("lora_a", "lora_b")) and p.requires_grad)
        raise RuntimeError(f"loss carries no gradient (flow.requires_grad={flow.requires_grad}, trainable adapter tensors={trainable}, "
                           f"checkpointing={grad_checkpoint})")
    return loss


@torch.no_grad()
def evaluate(pipeline, clip, modes=("correct", "wrong", "none"), seeds=(0, 1), loss_space="velocity"):
    """Mean loss per (mode, schedule step) on fixed noise draws."""
    result = {}
    for mode in modes:
        if mode == "wrong" and clip.wrong is None:
            continue
        per_step = []
        for step in range(len(SCHEDULE_INDEX)):
            values = []
            for seed in seeds:
                g = torch.Generator(device=clip.latent.device).manual_seed(1000 * step + seed)
                values.append(float(memory_loss(pipeline, clip, mode, step, g, loss_space, grad_checkpoint=False)))
            per_step.append(sum(values) / len(values))
        result[mode] = {"per_step": per_step, "mean": sum(per_step) / len(per_step)}
    return result
