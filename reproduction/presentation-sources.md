# Sources for the presentation video

`docs/demo/presentation.mp4` (125 s, 1920x1080, 30 fps) is rendered by `scripts/build_presentation.py` from
`reproduction/presentation-storyboard.json`. Captions only, no voice track: no offline text-to-speech was
installed, and no project text was sent to an outside service.

Re-render: `.venv-lingbot/bin/python scripts/build_presentation.py` (about one minute).
Visual check: `.venv-lingbot/bin/python scripts/presentation_contact_sheet.py docs/demo/presentation.mp4 <prefix> 3.4`.
The storyboard points at videos and result files in other branches' worktrees by absolute path
(`revisit-horizon`, `memory-v2`, `storage-ablation`, and the frozen final run's videos); those must be present to re-render.

| Section | What is shown | Source |
|---|---|---|
| Problem; Results side by side | PlayWorld scene GC026, scripted turn away / hold 12 s / turn back, seed 42, stock vs memory. **Selected example** (judge 2.4 vs 4.6), chosen among the largest judged gains and checked by eye. | Revisit-horizon benchmark: `outputs/revisit-horizon/rh-{stock,memory}-GC026-H12-seed42/rollout.mp4`; scores in `reproduction/revisit-horizon/analysis.json` (`per_episode`) |
| Benchmark | 171 tasks: 48 / 50 / 30 / 43 per dimension; published 2.12, 1.82, 1.14 | `external/PlayWorld/data/*/data.json`; published figures as quoted in `LEADERBOARD.md` and by the coordinator (Matrix-Game 1.14 not re-checked here) |
| Baseline failures | Stock videos GC038, IF013, OE007, OE036 from the frozen final run; each scored 1.0 on all three judge calls. Chosen by eye; the one-line failure descriptions are mine from watching the frames (OE036 also described in `docs/GALLERY.md`). "Most out-of-sight episodes score 1": 30 of 43 stock out-of-sight episodes are 1.0 on all three calls. | `reproduction/final-run/per-episode.csv`; videos `outputs/final1/stock-<split>-<task>-seed42/rollout.mp4` |
| "1.81–1.85 overall", baseline line | Stock overall 1.85 (full run) and 1.81 (final run, single call) vs published 1.82 | `reproduction/full-run/analysis.json`, `reproduction/final-run/analysis.json` (read at render time) |
| Method | 18 latent frames ≈ 4.5 s rolling cache (sink 6 + recent 12); archive of every latent frame with pose and DINO descriptor; pose-verified retrieval; recompute of 40-layer K/V for 10 retrieved frames; window sink 6 + retrieved 10 + recent 8 | `reproduction/memory-v2-config.json` (branch `memory-v2`) and the coordinator's method description; the diagrams are schematic |
| "≈ 6 MB per frame", "~200x" | 308,066,304 bytes of clean inputs for 50 stored latent frames = 6.2 MB per frame, against 1,235 MB per archived frame of exact keys and values | `outputs/memv2-step1/v2recompute-hold-GC007-seed42/run.json`; storage table |
| Chart | Judged score vs seconds away, means and 95% CIs, paired difference, wins/ties/losses; final at this render (160 of 160 scored): pooled over 6, 12, 24 s, per-horizon differences and the note that the gain shrinks at 24 s; small differences of 0.01 in interval ends against `docs/demo/horizon-summary.json` come from bootstrap resampling | `reproduction/revisit-horizon/analysis.json`, read at render time |
| Ablation | DINO revisit quality on GC007 turn-hold-return, seed 42: stock 0.35, exact 0.78, 8-bit 0.82, recompute 0.77. Only exact and 8-bit hold on a second scene; layer subsets, pooling, low-rank and int8 do not. Recompute: one scene, one seed. | `reproduction/memory-storage-ablation.md` and `memory-storage-ablation/table.md` (branch `storage-ablation`); `memory-v2-config.json` `primary_choice_evidence` |
| Real PlayWorld task | GC045, agent-driven, stock (final run) vs memory v2: judge 2.4 vs 4.0. **Selected example**; over the 36 revisit tasks ours scores lower (-0.35). | `reproduction/memory-v2-step2/per-case.csv` (branch `memory-v2`) |
| Limitations | Memory 1.80 vs stock 1.85 (full run); director 1.82 vs 1.81 (final run); judge SD 0.36; loss clip GC032 at 12 s (2.5 vs 1.0) | the two `analysis.json` files; `LEADERBOARD.md`; horizon `analysis.json` |
| Process | 3 audited defects; 12% of chunks are revisits; v2 about 200x smaller; 36 real revisit tasks, memory v2 final with three judge calls per video: -0.35 [-0.71, -0.01], 9 wins / 12 ties / 15 losses (also in Limitations and under the GC045 example) | `reproduction/INTEGRATION.md`, `reproduction/full-run-memory-audit.md`, `reproduction/full-run/analysis.json`; the -0.35 figure is the memory-v2 fork's final report relayed by the coordinator, not re-derived here (the single-call per-case CSV I read gives -0.12) |

The Process scene and the autoresearch slide also mention past.dev as the loop's findings memory (100 findings indexed; see `reproduction/research-memory/README.md`). It is not claimed to have influenced any result or the method.

Not verified: that the phase labels ("looking away", "camera turns back") match every frame (they are derived
from the scripted action list: 13 frames for the first chunk, 16 for each later one); the "K = 10 retrieved
frames" figure (taken from the coordinator's description).
