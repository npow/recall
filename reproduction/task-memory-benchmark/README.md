# Local task-matched memory pilot

The four-view memory method improves the two currently valid completed tasks: GC010 2.6→4.0 and held-out GC033 2.4→4.0 under the same local Claude PlayWorld rubric. Memory passes the released instruction gate on both; stock is partial. These are complete scripted 90-degree turn-and-return episodes, including an eight-second away-view hold, on identical H100 hardware and settings.

GC013 is quarantined because the released image is a coast while the caption/rubric describe a living room. Its raw2.4→3.2 scores and videos remain visible but are invalid as benchmark evidence. See ../input-audit-gc013.md and ../task-input-audit.json. The original unreviewed cohort summary is labeled original-unaudited-summary.json. The active cohort mean is null until replacement GC040 is generated and scored; missing or invalid cases do not silently disappear.

GC040 is declared before generation in the amended ../task-memory-benchmark-plan.json using a deterministic compatible-task rule. Its source image, and the sources for GC010/GC033, have been visually checked against their captions. No candidate selection uses its outputs.

The remaining limitations matter: sign lettering can blur, object counts can be uncertain and camera heading can drift. These three-task local scores use a scripted policy and Claude rather than the published adaptive Agent Player/Gemini setup. They are not comparable to the Genie 3 leaderboard score or a full 171-case overall. No training or image/output editing is used.

Complete videos, contacts, native manifests, comparison metrics, released gate/rubric contexts, judge responses and usage are retained per case. Hardware and encoder-offload settings are recorded. Both policies keep the same 18-frame attention budget; memory adds a bounded CPU key/value archive. Reproduce with `modal run --profile nissan-pow scripts/modal_lingbot_roundtrip.py --task-benchmark`, then `.venv-lingbot/bin/python scripts/evaluate_task_memory.py`.
