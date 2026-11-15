# WorldSync — two-minute presentation

**Use the [academic presentation cut](worldsync-academic-2min.mp4) for live presenting.** Exactly120seconds, silent,720p30fps. Sequence: problem → benchmark → four simultaneous failure videos → architecture → quantitative results → boardwalk comparison → pond comparison → memory comparison → process-failure comparison → limitations and future work. [Timed speaker notes](ACADEMIC-SPEAKER-NOTES.md). [Static slide backup](worldsync-academic-slides.pdf). [Duration and source verification](academic-verification.json).

Rebuild with `.venv-lingbot/bin/python scripts/build_academic_video.py`. All clips play at **normal speed**. Quadrants show the first16seconds; boardwalk and pond excerpts last12seconds each; the process-failure comparison lasts10seconds. The12-second memory segment joins source0–4seconds and23.8125–31.8125seconds, with the same waiting-period cut on both sides and an explicit label. Scores still use full originals. Held frames are labeled in the comparisons.

## Original demo cut


**Headline: +0.86 PlayWorld points on a preselected three-task instruction-following subset.**

- [Two-minute presentation video](worldsync-2min.mp4) — 1280×720, captioned, silent.
- [Standalone boardwalk side-by-side](IF022-side-by-side.mp4).
- [Voiceover / live presenter script](VOICEOVER.md).
- [Score slide](subset-scores.png).

| PlayWorld task | Legacy setup | WorldSync setup | Gain |
|---|---:|---:|---:|
| IF012 — walk toward the street lamp | 1.00 | 1.20 | +0.20 |
| IF006 — walk left into the pond | 2.00 | 2.56 | +0.56 |
| IF022 — continue along the boardwalk | 1.67 | 3.50 | +1.83 |
| **Mean** | **1.56** | **2.42** | **+0.86** |

Means use all three declared Gemini judgments for each original full video; displayed numbers are rounded. The exact values and original-video SHA256 hashes are in [evidence.json](evidence.json). Each pair uses the same LingBot-World2 14B causal-fast checkpoint, seed42 and GPU. We changed the complete controller/prompt/duration configuration, not the model weights. This is not a memory-only ablation.

The three task IDs were fixed before scoring. We chose to highlight the IF subset after reviewing the results. These are exploratory, single-seed results; other dimensions do not improve consistently, and no full-benchmark win is claimed. Boardwalk task completion remains partial. [All comparison results and raw judgments](../../reproduction/baseline-gap/ccv-snapshot/report.md).

## Video treatment

The boardwalk segment shows both complete episodes at2×. The legacy episode ends at7.8125seconds; its last frame is held with an explicit label while the WorldSync episode continues. The pond segment is a clearly labeled first16-second excerpt at1×; scores still come from the full original episodes. A late-frame artifact is shown on the closing slide. Preview resizing/compression is for presentation only and was not used for scoring.

The memory segment is a **separate** fixed-path experiment, not the configuration behind the IF scores. It shows both full31.8125-second videos at2.5×, then holds the final frames. Endpoint DINO similarity0.574→0.746 is a visual diagnostic, not a PlayWorld score. [Memory method and evidence](../../reproduction/memory-validation/README.md).

Regenerate from the retained source videos:

```bash
.venv-lingbot/bin/python scripts/build_hackathon_video.py
```

The renderer verifies original-video hashes against every IF judgment, derives the subset numbers from saved reports, and checks that the exported video contains exactly3600frames at30fps:120seconds. See [render verification](render-verification.json).
