# Differentiable memory for the LingBot fork

This is a proposed experiment, not an implemented improvement or a PlayWorld score. Keep the frozen baseline, pending text-conditioning confirmation and existing memory results separate.

## What the literature establishes

- [Neural Turing Machines](https://arxiv.org/abs/1410.5401) learn differentiable read/write addressing over an external memory. Their original algorithmic-task results do not establish compatibility with a pretrained video diffusion model.
- [End-To-End Memory Networks](https://arxiv.org/abs/1503.08895) use repeated attention reads over stored memories. This suggests a learned reader for our archived views, rather than replacing the video generator.
- [Titans](https://arxiv.org/abs/2501.00663) combines attention with a neural memory updated at inference. Its test-time updates still rely on a trained memory architecture; they do not provide an automatically usable interface to frozen LingBot features.
- [WorldMem](https://arxiv.org/abs/2504.12369) directly conditions video generation on memories associated with pose and time, including changing scene states.
- [World-T3](https://arxiv.org/abs/2610.17434), October 2026, consolidates scene features into fast weights using recursive least squares and reads them by pose. Its reported revisit PSNR improvement from 13.52 to 18.31 dB is not a PlayWorld result or proof that its memory transfers to LingBot.

The proposal below is our engineering inference from these mechanisms, not a reproduction of any of these papers.

## First experiment: learn the reader

Freeze the pretrained video model. Preserve its native sink and recent-frame attention. Store historical features with camera pose, timestamp and provenance. Train a small reader and a gated residual connection so the denoiser can use retrieved memories while keeping its original generative prior. A zero-initialized residual projection should reproduce the unmodified model before training.

Start with explicit memories rather than compressing everything into fast weights. This permits direct inspection of which observation supplied a detail, and isolates whether the model can use history before introducing compression error. A later fast-weight/RLS version must be compared against this explicit-memory control at matched compute and memory budgets.

The current implementation archives pre-RoPE normalized keys and native values at each layer. Those tensors are not automatically a suitable training target: their distributions depend on layer, noise level and conditioning. A learned reader must specify which clean insertion pass supplies its features and how those features enter attention. Averaging KV tensors from unrelated spatial patches is not a justified substitute for learning that interface.

## Training evidence required

Use separate, real or simulator-grounded leave-and-return trajectories with camera poses. During training, remove the target view from the recent context and provide the earlier corresponding observation only through memory. Include ordinary trajectories and moving-object examples so the adapter cannot improve revisit images by freezing the world. Never train on the benchmark's held-out episodes.

Use the model's native denoising objective on target chunks, with memory dropout and mismatched-memory controls. Verify that aligned history improves prediction over absent and shuffled history; decreasing training loss alone is insufficient. Backpropagation through frozen generator blocks may still be needed to train an early inserted adapter, so training cost must be measured rather than inferred from the adapter's parameter count.

## Acceptance and failure checks

Compare stock, our existing four-view retrieval, learned explicit memory and any compressed memory on the same held-out player-driven episodes. Freeze settings before judging. Report revisit consistency, actual commanded motion, dynamic-state persistence, inference latency, memory bytes, generation failures and the released PlayWorld dimension scores. Only a complete, documented four-dimension evaluation can support a claim against 2.12.

Inspect wrong-pose retrieval, moving objects that should not revert, repeated textures, generated errors written back as memories, unseen views and long periods without revisits. Commanded pose is not measured pose: our observed camera drift can make even a perfectly trained reader retrieve the wrong place. Learning memory alone also does not establish collision physics; the task-text pilot still walks characters through glass.

With the submission deadline approaching, do not displace a live confirmation or full evaluation with an unvalidated training run. First demonstrate memory dependence and a feasible training step on a small frozen-model probe, then decide whether an adapter run fits the remaining evaluation budget.
