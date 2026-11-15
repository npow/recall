# Recall demo (past.dev research memory)

Run 2026-11-15 10:37 PST. Of 100 findings sent, ingestion status at this time: {'completed': 66, 'processing': 34}. Recall below runs against whatever had completed; findings still processing can only add results later.
Text is verbatim recall output (top 3 per question, clipped to 420 chars). Dates are when each source write-up was last committed.
Earlier end-to-end check (a separate test point, since deleted): ingest -> status completed -> recall returned it with its source excerpt.

## Which memory result is still unconfirmed on a second scene?

1. [2026-11-15T17:38:42Z] (observation) A memory effect on judged scores is not established (outcome).
   - source: reproduction/INTEGRATION.md / Gemini rescoring versus the powered test (inconsistent, effect not established)

2. [2026-11-15T17:46:16Z] (observation) Memory v2 was positive on scripted paths (outcome).
   - source: LEADERBOARD.md / Memory v2 on real PlayWorld revisit tasks (null)

3. [2026-11-15T17:46:16Z] (source) WorldSync experiment finding from reproduction/revisit-horizon.md, section 'Where it still breaks'. It fades with time away. The paired gain is +1.31 at 6 s and +0.34 at 24 s. At 24 s object identity is near. the floor for both arms (1.03 against 1.26): after a long absence the scene is mostly not found again, with or. without this memory. Neither arm returns to the start view exactly. In the strips below the camera 
   - source: reproduction/revisit-horizon.md / Where it still breaks

## Where does the memory gain shrink with time away, and why?

1. [2026-11-15T17:46:16Z] (source) WorldSync experiment finding from reproduction/revisit-horizon.md, section 'Where it still breaks'. It fades with time away. The paired gain is +1.31 at 6 s and +0.34 at 24 s. At 24 s object identity is near. the floor for both arms (1.03 against 1.26): after a long absence the scene is mostly not found again, with or. without this memory. Neither arm returns to the start view exactly. In the strips below the camera 
   - source: reproduction/revisit-horizon.md / Where it still breaks

2. [2026-11-15T17:46:16Z] (claim) WorldSync revisit-horizon experiment absence length where gain shrinks: 24 seconds (exactly).
   - source: reproduction/revisit-horizon.md / Revisit horizon: how long can the camera look away?

3. [2026-11-15T17:46:16Z] (source) WorldSync experiment finding from reproduction/revisit-horizon.md, section 'Revisit horizon: how long can the camera look away?'. Revisit horizon: how long can the camera look away? Result. On 20 held-out scenes, archived-view memory raises the judged score at every absence length. we tested, with all four 95% intervals above zero. The gain is largest at 6 seconds away and shrinks by 24 seconds. It is a scripted-path
   - source: reproduction/revisit-horizon.md / Revisit horizon: how long can the camera look away?

## What did the audits find wrong with memory v1?

1. [2026-11-15T17:38:42Z] (observation) The independent audit confirmed three real defects in KV retention: selection is content-blind in early layers, other windows silently run stock attention, and the per-layer CPU KV archive is unbounded full-precision (problem).
   - source: reproduction/INTEGRATION.md / Audit notes that must travel with the code

2. [2026-11-15T17:38:42Z] (source) WorldSync experiment finding from reproduction/INTEGRATION.md, section 'Audit notes that must travel with the code'. KV retention (worldsync/lingbotkvretention.py): an independent audit confirmed three real defects. Selection is made once per. layer on the first, highest-noise attention call, so the query is content-blind in early layers; the gate hard-codes the. 18/6 window, so other windows silently run stock atten
   - source: reproduction/INTEGRATION.md / Audit notes that must travel with the code

3. [2026-11-15T17:46:16Z] (mention) Raised an issue: scenes worse with memory at 12 s.
   - source: reproduction/revisit-horizon.md / Where it still breaks

## What does stock LingBot-World2 score on the PlayWorld benchmark?

1. [2026-11-15T16:51:48Z] (claim) LingBot-World2 overall score: 1.82.
   - source: reproduction/final-run/table.md / secondary mean of 3

2. [2026-11-15T17:46:16Z] (source) WorldSync experiment finding from LEADERBOARD.md, section 'Published references — not reproduced here'. Model; Overall; Scope. Genie 3; 2.12; Published closed-model result. LingBot-World2; 1.82; Published hosted-service result; not our local checkpoint measurement. Source: pinned PlayWorld leaderboard. Overall averages GC, IF, in-sight evolution and out-of-sight evolution; the full dataset has 171 tasks (48/50/30/43)
   - source: LEADERBOARD.md / Published references — not reproduced here

3. [2026-11-15T16:51:48Z] (claim) LingBot-World2 IF score: 2.13.
   - source: reproduction/final-run/table.md / secondary mean of 3

## What happened when the stored memory was compressed?

1. [2026-11-15T17:29:36Z] (observation) The tests found nothing smaller than half storage reliable; achieving a large reduction requires a different design rather than compressing K/V (problem). Because Nothing smaller than half storage was reliable in these tests..
   - source: reproduction/memory-storage-ablation.md / Ranking and recommendation

2. [2026-11-15T11:30:01Z] (observation) The budget memory ran as coded, with no merge bug and no state leakage (outcome).
   - source: reproduction/full-run-memory-audit.md / Was budget memory actually working in the full 171-case run?

3. [2026-11-15T10:57:52Z] (observation) There was no gain where memory fired (outcome).
   - source: reproduction/full-run-memory-fired.md / Findings

