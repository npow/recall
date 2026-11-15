# Audited local task-matched memory pilot

Four-view historical memory improves all three valid tasks under the same local Claude PlayWorld rubric. These are complete scripted 90-degree turn-and-return episodes, including an eight-second away-view hold, on identical H100 hardware and settings.

| Case | Split | Stock | Memory | Instruction gate, stock → memory |
|---|---|---:|---:|---|
| GC010, western street | Development | 2.6 | 4.0 | partial → pass |
| GC033, display window | Held out | 2.4 | 4.0 | partial → pass |
| GC040, beach tents | Audited replacement development case | 2.4 | 2.7 | partial → pass |
| GC013 | Quarantined invalid source pairing | raw 2.4 | raw 3.2 | Excluded explicitly from valid-input means |

The amended valid-input cohort mean is **2.47 stock versus 3.57 memory** (+1.10 on the local five-point rubric). No uncertainty estimate is available from this small cohort. The generator keeps the same 18-frame attention budget, official checkpoint, sampler and poses within each pair. Complete native-camera audits and exact pre-intervention raw-pixel prefixes pass. No training, image editing or output-frame pasting is used.

GC013 is invalid because the released image is a coast while its caption/rubric describe a living room. The image matches the official pinned bytes. Its raw videos, scores and original unaudited summary remain visible, but do not establish benchmark quality. See ../input-audit-gc013.md and ../task-input-audit.json. GC040 was declared before generation using the lowest remaining compatible development task ID, and visually audited against its source caption. The amended cohort has four attempted case IDs, including the quarantined case; invalid and missing cases do not disappear silently.

Failures remain. Western-street lettering blurs and a porch-person count is uncertain. The display-window paper text loses legibility. On the beach, the sky/sea change from overcast to clear, an extra tent and a large yellow object appear, and the final layout is imperfect; its 2.7 score is only moderate. Exact heading and dynamic-process consistency are not solved.

The policy is scripted, and Claude replaces the published adaptive Agent Player/Gemini setup. These local subset scores are not comparable to Genie 3's leaderboard result or a full 171-case overall. Both methods use the same generic judge label to avoid exposing the memory treatment.

Videos, contact sheets, endpoint images, native manifests, comparison metrics, released gate/rubric contexts, full judge responses and usage are retained per case. Hardware and text-encoder offload settings are recorded. Memory adds a bounded 19.765 GB CPU archive. Reproduce with `modal run --profile nissan-pow scripts/modal_lingbot_roundtrip.py --task-benchmark`, then `.venv-lingbot/bin/python scripts/evaluate_task_memory.py`.
