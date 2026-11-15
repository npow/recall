The existing adaptive player can be reused without changing its request, response validation or camera controller. Keep the small checkpoint adapter outside the normal 14B entrypoint. The current small native probe already verifies strict compatibility with source 1895d300, actual 1.7095B DiT parameters, shift 5 and four original ODE steps; T5/VAE assets remain shared and separately identified.

The minimal implementation is a small adapter context around rollout.main: reuse the existing isolated backend/builder from worldsync/lingbot_small_probe.py, restore the constructor/loader/schedule on exit, and pass ordinary adaptive arguments rather than fixed actions. Give these runs a distinct small-model protocol/model identity. Do not enable fourteen-billion LoRA/retrieval adapters accidentally. The player implementation, image observation, action schema, turn speed and stopping challenges should remain unchanged between small-model arms.

| Profile | Native cache | BF16 native K/V allocation at 832x464 |
|---|---|---:|
| Own baseline |18 latent frames,6sinks|5.003GB|
| Longer context |70 latent frames,6sinks|19.457GB|
| Native global,40chunk budget |160latent frames, no eviction|44.473GB|
| Native global,60chunk budget |240latent frames, no eviction|66.709GB|

These figures are only native K/V storage: add 3.419GB DiT weights, VAE, T5 lifecycle, activations, allocator reserve, CPU payloads and the 16GiB physical reserve. Extrapolating the observed 70-frame GPU peak gives roughly 57GB for full 40-chunk global context and 79GB for 60 chunks, before uncertainty. They are not measured budgets. The previous small replay's 45GB application estimate cannot be reused for the global profiles.

The unmodified stock knob resolver rejects -1. The isolated wrapper should validate all normal arguments first, then translate an explicitly declared native-global profile into the small builder's native -1 sentinel and report it honestly. Restore that wrapper-local resolver on exit; do not weaken the fourteen-billion CLI. Seventy-frame adaptive mode can use the existing hillclimbing gate. Global allocation is bounded by the declared episode length, rather than an infinite memory.

CPU native attention tests show global and 70-frame modes are bit-exact for 52 generated latent frames:65 forward passes, including repeated denoising writes, in both BF16 and FP32. This equivalence stops being a justified assumption once an episode exceeds the 70-frame window. GPU pixel parity and adaptive behavior are not yet tested.

First run a source-disjoint short adaptive pair through the same player and inspect actual actions, camera motion and termination. Then test a genuine long revisit before considering a larger evaluation. The fixed 13-chunk score difference is partial visual-rubric credit without a completed revisit; it does not establish a world-memory improvement. No GPU launch or large implementation is part of this assessment.
