# GC013 released asset mismatch and quarantine

The released GC013 image is a coastline with a seagull. Its caption and rubric instead describe a living room containing a sofa, coffee table, art wall and bookshelf. The generator changes from the coast to that captioned room at the beginning, before retrieval starts. This is an input integrity problem and must not be represented as a valid benchmark improvement.

The local image is byte-identical to the current pinned Hugging Face asset: SHA-256 2f16ab868ee68eb2c111b71aedefe171716f883baf305aad3e4be5f2130a9f57,1,810,006 bytes. A fresh HTTP download from the official pinned resolve URL matches it. The current dataset revision still equals our pinned 697e177194a1c4d8afe9068756fe743ac23ee1e8. Therefore no local asset repair is justified from available release provenance. We do not substitute an arbitrary living-room photo or rewrite the released rubric.

Sources: [released image](https://huggingface.co/datasets/jocelynd/playworld-bench/blob/697e177194a1c4d8afe9068756fe743ac23ee1e8/gc/images/GC013.jpg), [released metadata](https://huggingface.co/datasets/jocelynd/playworld-bench/blob/697e177194a1c4d8afe9068756fe743ac23ee1e8/gc/data.json).

Both original videos and their raw scores(stock 2.4,memory 3.2) are retained under task-memory-benchmark/GC013-seed42-turn90left_wait_return/. The original unaudited three-case summary is preserved explicitly as unaudited. The active evaluator marks GC013 invalid, retains it as a visible quarantined row and suppresses a valid cohort mean until all valid declared pairs are complete.

GC040 is the replacement, selected before its generation as the lowest remaining task ID with a compatible 90-degree task in the frozen development split. Its beach-tent source image was visually checked against caption and rubric. Existing GC010 and held-out GC033 source images were also inspected and match. The input audit records image and caption hashes, and generation now checks them before allocating model work. No memory settings or evaluation criteria change based on the replacement's results.

The amended plan retains the original case and declares GC040 separately. This is an amended local pilot with a documented source-data exception, not an unmodified published benchmark reproduction. Byte verification of all 171 assets remains true; complete semantic verification does not. No external issue or message has been sent to dataset maintainers.
