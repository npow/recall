"""Write docs/DEMO.md and docs/demo/APPENDIX.md from docs/demo/horizon-summary.json and optional memory-v2 step-2 numbers.

Run scripts/build_demo.py first. Observations on each clip were written after viewing the strips.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "docs/demo"

SEEN = {
    ("GC011", 12): "Stock comes back to a different wall and a porch railing; with memory the front door, sidelights and console table of the start view are back.",
    ("GC034", 12): "Stock comes back to a close-up through a fence with a different layout; with memory the stone gate and the wooden hut beside it are back.",
    ("GC026", 12): "Stock comes back to a close-up of a different shop door; with memory the red Diamond Design awning and arched entrance are back, seen from further away.",
    ("GC037", 12): "Stock comes back to a wall with a clock that was never there; with memory the dark window and counter of the start view are back, framed differently.",
    ("GC025", 12): "Stock ends in a different bedroom; with memory the hallway painting from the start view is back. The two judge calls disagree (5.0 and 1.0).",
    ("GC021", 24): "Neither arm returns to the Eiffel Tower view. With memory the single golden statue and colonnade stay consistent; stock adds extra statues and graffiti. Not a visible return.",
    ("GC032", 12): "A loss. Neither arm returns to the bridge view; with memory the final frames show a distorted blue railing.",
    ("GC002", 12): "A loss. Neither arm returns to the central gallery view; stock's final view is the more coherent one.",
}
FEATURED = [("GC011", 12), ("GC034", 12), ("GC026", 12)]


def interval(triple, digits=2):
    return f"{triple[0]:+.{digits}f} [{triple[1]:+.{digits}f}, {triple[2]:+.{digits}f}]"


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    summary = json.loads((DEMO / "horizon-summary.json").read_text())
    interim = summary["interim"]
    tag = " (interim)" if interim else ""
    horizons = {int(k): v for k, v in summary["horizons"].items() if v["n"]}
    pairs = {(p["task"], p["horizon"]): p for p in summary["pairs_h12_h24"]}
    h12 = horizons[12]
    pooled = summary["pooled_h12_h24_by_scene"]
    rows = []
    for horizon, row in sorted(horizons.items()):
        rows.append(f"| {horizon} s | {row['n']} | {row['stock'][0]:.2f} | {row['memory'][0]:.2f} | {interval(row['delta'])} | "
                    f"{row['wins']}/{row['ties']}/{row['losses']} |")
    memv2_line, memv2_limit = "", ""
    step2 = DEMO / "memv2-step2-summary.json"
    if step2.is_file():
        data = json.loads(step2.read_text())
        everything, held = data["all"], data["held_out"]
        memv2_line = (f"**Memory v2 on PlayWorld's own revisit tasks** (pre-registered: the {everything['pairs']} geometry and out-of-sight "
                      f"tasks whose path revisits a view, `docs/demo/sources/memory-v2-config.json`): no gain. Paired judged difference "
                      f"{interval(everything['delta'])}, {everything['wins']}/{everything['ties']}/{everything['losses']} wins/ties/losses; "
                      f"held-out {held['pairs']} tasks {interval(held['delta'])} (`docs/demo/sources/memory-v2-step2-per-case.csv`). "
                      "One selected example from that set shows what it does when it fires:\n\n"
                      "**GC045, \"move 10 steps left, then 10 steps right to return\": judged 2.4 → 4.0** (selected example; a development task). "
                      "The player took the same 9 actions in both arms and the first 5 chunks are pixel-identical; memory fired once, at the turn-around. "
                      "On the way back, stock pushes a large red traffic light and a changed street sign into the frame; with memory v2 the "
                      "Pleasure Pier arch stays framed as it was. [video](demo/playworld-GC045.mp4)\n\n"
                      "![GC045 freeze frame](demo/playworld-GC045-freeze.jpg)\n\n![GC045](demo/playworld-GC045.jpg)")
        memv2_limit = (f"- **Memory v2 does not raise scores on PlayWorld's own revisit tasks**: {interval(everything['delta'])} over "
                       f"{everything['pairs']} tasks. It usually fires on only 1 to 5 chunks per episode, often at or after the return itself, and "
                       "single judge calls and different player endings swing individual tasks by several points.\n")
    examples = ""
    for task, horizon in FEATURED:
        pair = pairs.get((task, horizon))
        if not pair:
            continue
        name = f"horizon-win-{task}-H{horizon}"
        examples += (f"**{task}, {horizon} s away: judged {pair['stock']:.1f} → {pair['memory']:.1f}.** {SEEN[(task, horizon)]} "
                     f"[video](demo/{name}.mp4)\n\n![{task}](demo/{name}.jpg)\n\n")
    page = f"""# WorldSync demo: making a generated world remember where it has been

**Track 4 question: how long can a generated world hold together while an agent plays in it?**

We took an open world model (LingBot-World2, 14B) and added a **training-free episodic memory**: when the camera comes back to
a place it has already seen, the model is handed its own earlier frames of that place. It stores only the frame and recomputes
the model's internal state on demand, which takes **about 200× less memory** than storing that state.

![hero freeze frame](demo/hero.jpg)

**Hero video (20 s): [demo/hero.mp4](demo/hero.mp4)** · selected example, PlayWorld scene GC011: the camera looks away for
12 s and turns back. Left: LingBot-World2 as released. Right: with our memory. Alternates: [GC034](demo/hero-GC034.mp4), [GC026](demo/hero-GC026.mp4).

| | Headline result | Status |
|---|---|---|
| 1 | **After 12 s looking away, on 20 unseen PlayWorld scenes: judged consistency {h12['stock'][0]:.2f} → {h12['memory'][0]:.2f}, paired {interval(h12['delta'])}, {h12['wins']} wins / {h12['ties']} ties / {h12['losses']} losses** | pre-registered{tag} |
| 2 | **6.2 MB per remembered frame instead of 1,235 MB (about 200×) at the same recall** (0.77 against 0.78; stock 0.35) | one scene, one seed |
| 3 | **Published baseline reproduced twice with the open checkpoint: 1.85 and 1.81 overall against the published 1.82** | all 171 PlayWorld tasks |
| 4 | **Best subset of PlayWorld's own tasks: out-of-sight evolution with our text director, 1.32 → 1.53, a trend of +0.20 [−0.12, +0.58]** | post-hoc choice of subset and metric |

One honest line: on the full PlayWorld benchmark our methods score 1.80 to 1.86 overall, level with LingBot's published 1.82
and below Genie 3's 2.12, and no subset of PlayWorld's own tasks shows a gain whose interval excludes zero. The clear gain is on
long absences, which PlayWorld's own tasks rarely contain.

---

## Results

### How long does the world hold together?

![judged score against seconds away](demo/fig1-horizon.png)

Twenty PlayWorld geometry-consistency scenes that we never used in development, chosen by a hash of the scene id. The camera
turns away 180°, waits, and turns back. Same seed and same GPU for both arms. The Gemini judge scores each video twice with
PlayWorld's released consistency rubric. These are PlayWorld scenes with our own scripted path, not PlayWorld's task list.

| seconds away | scenes | as released | with our memory | paired difference [95% CI] | wins/ties/losses |
|---:|---:|---:|---:|---|---|
{chr(10).join(rows)}

Pooled over the 12 s and 24 s absences, scenes as units: {interval(pooled['delta'])} (n = {pooled['n']}){tag}.

### Selected examples

Each clip shows the whole episode, sped up to about 10 s. The full rule-based set, including losses, is in [demo/APPENDIX.md](demo/APPENDIX.md).

{examples}### About 200× less memory

![memory footprint](demo/fig2-footprint.png)

To reuse an old frame, the model needs that frame's internal state at all 40 of its layers: 1,235 MB per frame. Memory v2 keeps
only the frame, 6.2 MB with its conditioning, and recomputes the state when the camera returns: 13.5 s of extra compute in a
14-chunk episode, GPU memory 72.4 GB against 68.8 GB for the stock model.
[Side-by-side video, GC007](demo/memv2-hold-GC007.mp4): stock ends at a storefront close-up; memory v2 ends on the facade and plaza seen at the start.

![memory v2 on GC007](demo/memv2-hold-GC007.jpg)

### On PlayWorld's own tasks

| 171 tasks, Gemini judge | overall [95% CI] |
|---|---|
| LingBot-World2, published (hosted service) | 1.82 |
| LingBot-World2 open checkpoint, our run 1 | 1.85 [1.67, 2.04] |
| LingBot-World2 open checkpoint, our run 2 | 1.81 [1.64, 1.99] |
| + our text director (run 2 hardware) | 1.82 [1.66, 2.01]; mean of 3 judge calls: 1.86 [1.70, 2.02] |
| Genie 3, published | 2.12 |

The text director writes the world model's prompt in the event-caption format it was trained on. Across all tasks it is a small
tendency (+0.07 [−0.06, +0.22] with three judge calls). Its best dimension, chosen after seeing the results, is out-of-sight
evolution on held-out tasks: 1.32 → 1.53, +0.20 [−0.12, +0.58], 9 wins / 19 ties / 9 losses. Selected examples (more, with
losses, in [GALLERY.md](GALLERY.md)):

**OE036: judged 1.0 → 5.0.** After the camera tilts up and returns, stock has replaced the man and dog; with the director they keep walking away. [video](gallery/outsight-OE036.mp4)

![OE036](gallery/outsight-OE036.jpg)

**OE030: judged 1.0 → 4.25.** Stock loses the ramp and the car; the director keeps the same car on the same ramp. [video](gallery/outsight-OE030.mp4)

![OE030](gallery/outsight-OE030.jpg)

{memv2_line}

### Run the demo

One side-by-side on one 80 GB GPU (about 4 minutes per arm). `scripts/prepare_reproduction.py` fetches the PlayWorld inputs;
the LingBot checkpoint and environment are described in `reproduction/lingbot.md`.

```bash
ACTIONS="WAIT RIGHT RIGHT RIGHT $(printf 'WAIT %.0s' {{1..12}})LEFT LEFT LEFT WAIT"
COMMON="--split gc --task-id GC011 --seed 42 --max-chunks 20 --native-fixed-controls --fixed-actions $ACTIONS \\
  --turn-degrees-per-second 68 --offload-text-encoder"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m worldsync.lingbot_rollout $COMMON --output outputs/demo/stock
python -m worldsync.lingbot_rollout $COMMON --budget-memory --budget-rope native --budget-retrieved 10 --output outputs/demo/memory
ffmpeg -i outputs/demo/stock/rollout.mp4 -i outputs/demo/memory/rollout.mp4 -filter_complex hstack outputs/demo/side-by-side.mp4
```

Add `--dry-run` to check the arguments without a GPU (verified on `main`). The benchmark videos above were generated with the
same flags at commit `edb2b83` on Lambda H100s; pixels differ between GPU models, so compare arms on one GPU.

---

## Research rigour

- **A clear question with a baseline first.** We reproduced the published LingBot-World2 score on all 171 PlayWorld tasks before
  claiming anything, twice, on different hardware: 1.85 and 1.81 against 1.82 (`reproduction/full-run/table.md`, `reproduction/final-run/table.md`).
- **Frozen before generation.** Every comparison has a config committed before its first video: `reproduction/full-run-config.json`,
  `reproduction/final-run-config.json`, `docs/demo/sources/revisit-horizon-config.json`, `docs/demo/sources/memory-v2-config.json`,
  and a success criterion written before any full score existed (`reproduction/pass-criterion.md`).
- **Paired designs with intervals.** Same task, same seed, same GPU model for both arms; bootstrap 95% intervals over tasks;
  development scenes excluded from held-out results; wins, ties and losses reported next to every mean.
- **Ablations.** Nineteen storage formats for the memory (`docs/demo/sources/memory-storage-ablation-table.md`): 8-bit keeps
  the quality at half the size, no subset of layers does, pooled summaries do not, low-rank at rank 64 looks promising in one
  run. Retrieval rules v1, v2 and content-only side by side (`docs/demo/sources/memory-v2-step1-table.md`).
- **Independent audits.** A five-lens audit of baseline fidelity with a skeptic per claim (`reproduction/baseline-fidelity-audit.md`),
  a code-and-trace audit of the memory (`reproduction/full-run-memory-audit.md`), and a judge-repeat study that measured the
  judge's own noise at 0.36 points per video (`reproduction/judge-repeat/README.md`).

## Novelty

LingBot-World2 keeps a rolling cache of about 18 latent frames (6 anchor frames plus the most recent 12), roughly 4.5 s of
video. Anything older is gone, so a view the camera left is re-invented when it returns. Our memory adds four pieces, with no training:

1. **An episodic archive of every latent frame**, with its camera pose and a visual descriptor. Nothing is evicted.
2. **Pose-verified retrieval.** A stored frame is used only if its pose matches where the camera is going and the place is
   verified: either only rotation has happened since, or the current view matches a stored view of that place.
3. **Recompute and splice.** The keys and values of the retrieved frames are recomputed at all 40 layers with one clean pass,
   then spliced into the attention window in place of part of the recent history, keeping the same attention budget.
4. **Inert when not needed.** When no stored view qualifies, the output is bit-identical to the stock model.

How it differs from published work, as we read it (`reproduction/literature-review-2026-11-14.json`, `reproduction/vmem-lsm-robustdreamer.md`):

| | stores | chooses by | training |
|---|---|---|---|
| WorldKV | evicted keys and values | relevance of cached tokens | none |
| DeCoPrune | pruned keys and values | denoising discrepancy per token | none |
| VMem | past views, indexed by 3D surfels | what is visible from the new camera | fine-tuned generator |
| **ours (memory v2)** | **the frame only; state recomputed** | **pose, verified by content** | **none** |

What the tests of memory v2 establish (`docs/demo/sources/memory-v2-step1-table.md`): it is pixel-identical to the stock model
whenever it does not fire (v1 also retrieved on 338 of 726 chunks, 47%, where nothing was being revisited); recomputed state matches state
stored on disk (0.77 against 0.82 on the hold path, 0.62 against 0.63 on the slow turn); it needs no archive of internal state;
and it removes v1's harm on the slow 360° turn.

Storing the frame and recomputing the state is what gives the 200× saving; we did not find it in the papers we surveyed, and
we have not searched exhaustively. We also built 8-bit and low-rank variants of the stored state as ablations.

## Autonomy and self-improvement

![the loop](demo/fig3-loop.png)

- **Most of the loop ran without a human.** Agent forks designed experiments, wrote the code, launched up to 14 cloud GPUs,
  scored videos with the judge, analysed the results and pushed them, in parallel, for about 20 hours.
- **It checked itself.** An adversarial multi-agent audit (one reviewer per method, one skeptic per claimed bug) found three real
  bugs in our first retrieval code: a query computed from pure noise, a hard-coded window size, an unbounded archive. Memory v1 fixed them.
- **It learned from a null.** Memory v1 gave no gain on the full benchmark (1.80 against 1.85). A second audit and an offline
  replay showed why: only 12% of chunks are real revisits, v1 also retrieved on 338 chunks where nothing was revisited, and
  its 24-frame archive forgot after about 6 s.
- **The fix measurably improved.** Memory v2 retrieves on 108 of 115 real revisits and on none of the others. On a slow 360°
  turn v1 scored below stock (0.49 against 0.57); v2 scores 0.63. Storage fell from 1,235 MB to 6.2 MB per frame.
- **The human's role, accurately.** The human set the goals, approved budgets, supplied credentials, and made a handful of
  decisions. Two design ideas came from the human: store the frame and recompute the rest, and keep the archive on disk. The
  agents implemented, validated and compared them, and reported that a third suggestion (retrieve by content alone) scored worse.

## Interpretability and reproducibility

- **Every number has a file.** Per-task scores: `reproduction/full-run/per-episode.csv`, `reproduction/final-run/per-episode.csv`;
  the judge's full output for each video: `reproduction/final-run/gemini-call1/`; horizon benchmark pairs: `docs/demo/horizon-summary.json`.
- **The leaderboard is generated, not typed**: `scripts/update_leaderboard.py` writes [LEADERBOARD.md](../LEADERBOARD.md) from result files.
- **Decisions are logged.** Run chronologies with every deviation (`reproduction/full-run-log.md`), cloud instances and spend
  (`reproduction/lambda-log.md`), the note that coordinated three agent sessions (`reproduction/COORDINATION.md`), and what
  each experiment found, positive or negative (`reproduction/INTEGRATION.md`).
- **Code runs.** The flags are default-off options of one rollout script; the test suite passes; the command block above
  reproduces a side-by-side. Memory v2 and the storage variants live on branches `memory-v2` and `storage-ablation`.
- **Failures are reported.** See the next section, [GALLERY.md](GALLERY.md) for losses shown next to wins, and
  [docs/METHOD.md](METHOD.md) for the methods that did not work.

## Limits and what did not work

- **PlayWorld overall is unchanged.** Memory v1 1.80 against stock 1.85; director 1.82 against 1.81. Neither is above Genie 3's 2.12, and the pre-registered criterion was not met.
- **PlayWorld's tasks rarely need memory.** Only 12% of geometry and out-of-sight chunks revisit a view the stock model had forgotten.
- **The gain is in what the judge sees, not in our own similarity measure.** On the 20 unseen scenes, final-frame similarity to the start view is {h12['dino_stock']:.2f} against {h12['dino_memory']:.2f}: the judge credits consistent objects and layout more than an exact return.
{memv2_limit}- **The horizon benchmark tested memory v1**, because v2 had not passed its checks when the design was frozen. The judge is also noisy (0.36 points on the same video), which is why the tables report intervals.
- **Forward-and-back moves are not undone by the model**, and memory cannot fix that. A fine-tuned memory adapter (48 training pairs) did not help and hurt ordinary generation.

## What we would do next

- **A trained memory layer**: a small adapter that reads a compact summary of past views, in place of substituting state at all 40 layers.
- **Retrieval by what is visible**, as in VMem, so forward-and-back returns pick the right frames.
- **Longer tasks**: a benchmark split where the agent is away for 10 to 30 seconds, which is where worlds fall apart.

## Demo script (2 minutes)

1. **The problem.** Play the left half of the hero video. The model looks away for 12 seconds, turns back, and the room it left is gone. It only remembers about 4.5 seconds.
2. **The fix.** Play the right half. We hand the model its own earlier frames of the place it is returning to. No training. The front door is back.
3. **It holds on scenes we never tuned on.** Show Figure 1. On 20 unseen scenes the judged score goes from {h12['stock'][0]:.2f} to {h12['memory'][0]:.2f} after 12 seconds away, {h12['wins']} wins to {h12['losses']} losses.
4. **It is cheap, and it improved itself.** Show Figures 2 and 3. Agents audited the first version, found why it failed on real tasks, and built one that keeps the frame, not the model's state: 6 MB instead of 1.2 GB.
5. **Where we stand, honestly.** We reproduced the published baseline twice. On PlayWorld's own tasks we are level with it, not above, because those tasks rarely look away for long. Long absences are where memory pays.

Assets are built by `scripts/build_demo.py` and this page by `scripts/write_demo_page.py`.
"""
    (ROOT / "docs/DEMO.md").write_text(page)
    appendix = ["# Demo appendix: the full rule-based set", "",
                f"Rule: the largest paired judged differences at 12 s or 24 s away, one entry per scene, plus the most negative ones{tag}. "
                "Every pair is in `horizon-summary.json`. Observations were written after viewing each strip.", "",
                "| scene | away | as released | with memory | difference | judge calls (released / memory) | final-frame similarity |",
                "|---|---:|---:|---:|---:|---|---|"]
    for pair in summary["pairs_h12_h24"]:
        appendix.append(f"| {pair['task']} | {pair['horizon']} s | {pair['stock']:.2f} | {pair['memory']:.2f} | {pair['delta']:+.2f} | "
                        f"{pair['stock_calls']} / {pair['memory_calls']} | {pair['dino_stock']:.2f} / {pair['dino_memory']:.2f} |")
    appendix += ["", "## Clips", ""]
    for (task, horizon), text in SEEN.items():
        pair = pairs.get((task, horizon))
        kind = "win" if pair and pair["delta"] > 0 else "loss"
        name = f"horizon-{kind}-{task}-H{horizon}"
        if pair and (DEMO / f"{name}.jpg").is_file():
            appendix += [f"**{task}, {horizon} s away: {pair['stock']:.1f} → {pair['memory']:.1f}.** {text} [video]({name}.mp4)", "",
                         f"![{task}]({name}.jpg)", ""]
    if (DEMO / "playworld-OE044.jpg").is_file():
        appendix += ["## A loss on a real PlayWorld task (memory v2, OE044)", "",
                     "Judged 3.0 → 1.0. Stock ends with the plate and the person still at the counter; memory v2 ends on a sink view "
                     "without them. Memory fired on 9 of 13 chunks here. [video](playworld-OE044.mp4)", "", "![OE044](playworld-OE044.jpg)", ""]
    appendix += ["## Memory v2 on a slow 360° turn (GC007)", "",
                 "Both arms end on a similar plaza view, not the start view; the similarity scores are 0.57 (stock) and 0.62 (v2 recompute; 0.63 for v2 with the disk archive). "
                 "The point of this path is that v2 no longer scores below stock, as the first version did (0.49).", "",
                 "![slow 360](memv2-slow360-GC007.jpg)", ""]
    (DEMO / "APPENDIX.md").write_text("\n".join(appendix))


if __name__ == "__main__":
    main()
