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
    parser.add_argument("--memv2-report", type=Path)
    parser.add_argument("--memv2-final", action="store_true")
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
    memv2_line = ("Memory v2 on the 36 PlayWorld tasks that contain a real revisit (pre-registered in "
                  "`docs/demo/sources/memory-v2-config.json`): still being scored when this page was written.")
    if args.memv2_report and args.memv2_report.is_file():
        judge = json.loads(args.memv2_report.read_text())["summary"]["judge_delta"]["all"]
        mean, low, high = judge["mean_ci95"]
        wins, ties, losses = judge["wins_ties_losses"]
        state = "final" if args.memv2_final else "interim"
        memv2_line = (f"Memory v2 on the PlayWorld tasks that contain a real revisit (pre-registered, {state}, {judge['pairs']} pairs): "
                      f"paired judged difference {mean:+.2f} [{low:+.2f}, {high:+.2f}], {wins}/{ties}/{losses} wins/ties/losses. "
                      f"{'No gain on real episodes.' if low <= 0 else 'A gain on real episodes.'}")
    page = f"""# WorldSync demo: making a generated world remember where it has been

**Track 4 question: how long can a generated world hold together while an agent plays in it?**

We took an open world model (LingBot-World2, 14B) and added a **training-free memory**: when the camera comes back to a place
it has already seen, the model is handed its own earlier frames of that place. It stores only the frame and recomputes the
model's internal state on demand, which takes **about 200× less memory** than storing that state.

![hero freeze frame](demo/hero.jpg)

**Hero video (20 s): [demo/hero.mp4](demo/hero.mp4)** · selected example, PlayWorld scene GC011, camera looks away for 12 s and turns back.
Left: LingBot-World2 as released. Right: with our memory.

## Headline results

| | Result | Status |
|---|---|---|
| 1 | **Looking away for 12 s and turning back, 20 unseen PlayWorld scenes: judged consistency {h12['stock'][0]:.2f} → {h12['memory'][0]:.2f}, paired {interval(h12['delta'])}, {h12['wins']} wins / {h12['ties']} ties / {h12['losses']} losses** | pre-registered{tag} |
| 2 | **Memory footprint: 6.2 MB per remembered frame instead of 1,235 MB (about 200×), with the same recall** (0.77 vs 0.78 on the test path; stock 0.35) | one scene, one seed |
| 3 | **Published baseline reproduced twice with the open checkpoint: 1.85 and 1.81 overall, against the published 1.82** (171 PlayWorld tasks each) | full benchmark |
| 4 | **On PlayWorld's own tasks, the best subset is out-of-sight evolution with our text director: 1.32 → 1.53, a trend of +0.20 [−0.12, +0.58]** (37 held-out tasks, mean of 3 judge calls) | post-hoc choice of subset and metric; interval includes zero |

One honest line: on the full PlayWorld benchmark our methods score 1.80 to 1.86 overall, level with LingBot's published 1.82
and below Genie 3's 2.12, and no subset of PlayWorld's own tasks shows a gain whose interval excludes zero. The clear gain is on
long absences, which PlayWorld's own tasks rarely contain.

## 1. How long does the world hold together?

![judged score against seconds away](demo/fig1-horizon.png)

Twenty PlayWorld geometry-consistency scenes that we never used in development, chosen by a hash of the scene id. The camera
turns away 180°, waits, and turns back. Same seed and same GPU for both arms. The Gemini judge scores each video twice with
PlayWorld's released consistency rubric. The design was frozen before any video was generated
(`docs/demo/sources/revisit-horizon-config.json`). These are PlayWorld scenes with our own scripted
path, not PlayWorld's task list.

| seconds away | scenes | as released | with our memory | paired difference [95% CI] | wins/ties/losses |
|---:|---:|---:|---:|---|---|
{chr(10).join(rows)}

Pooled over the 12 s and 24 s absences, scenes as units: {interval(pooled['delta'])} (n = {pooled['n']}){tag}.

### Selected examples

Each clip shows the whole episode, sped up to about 10 s. Full rule-based set, including losses: [demo/APPENDIX.md](demo/APPENDIX.md).

"""
    for task, horizon in FEATURED:
        pair = pairs.get((task, horizon))
        if not pair:
            continue
        name = f"horizon-win-{task}-H{horizon}"
        page += (f"**{task}, {horizon} s away: judged {pair['stock']:.1f} → {pair['memory']:.1f}.** {SEEN[(task, horizon)]} "
                 f"[video](demo/{name}.mp4)\n\n![{task}](demo/{name}.jpg)\n\n")
    page += f"""## 2. About 200× less memory

![memory footprint](demo/fig2-footprint.png)

To reuse an old frame, the model needs that frame's internal state at all 40 of its layers: 1,235 MB per frame. Our first
version stored that. Compressing it does not get far: 8-bit storage halves it, and keeping only some layers or a summary loses
most of the effect (`docs/demo/sources/memory-storage-ablation-table.md`). Memory v2 keeps only the
frame itself, 6.2 MB with its conditioning, and recomputes the internal state when the camera returns: 13.5 s of extra compute
in a 14-chunk episode, GPU memory 72.4 GB against 68.8 GB for the stock model.

It also acts only when it should. It retrieves when the camera really is back at a stored view, and is pixel-identical to the
stock model everywhere else. On a slow 360° turn, where our first version scored below stock (0.49 against 0.57), v2 scores
0.62 (`docs/demo/sources/memory-v2-step1-table.md`).

[Side-by-side video, GC007 turn / hold / return](demo/memv2-hold-GC007.mp4)

![memory v2 on GC007](demo/memv2-hold-GC007.jpg)

Stock ends at a storefront close-up; memory v2 ends on the Primemall facade and plaza seen at the start.

## 3. The baseline, reproduced

We ran all 171 PlayWorld tasks with the open LingBot-World2 checkpoint, an agent player and the released Gemini rubric, twice
on different hardware: **1.85 [1.67, 2.04]** and **1.81 [1.64, 1.99]** overall, against the published **1.82**
(`reproduction/full-run/table.md`, `reproduction/final-run/table.md`). An independent audit found no pipeline bug
(`reproduction/baseline-fidelity-audit.md`).

## 4. On PlayWorld's own tasks

We also tried a text "director" that writes the world model's prompt in the event-caption format it was trained on. Across all
171 tasks it is a small tendency, not an established gain: 1.81 → 1.82 with one judge call (+0.02 [−0.15, +0.19]) and
1.78 → 1.86 averaged over three (+0.07 [−0.06, +0.22]). Its best dimension, chosen after seeing the results, is out-of-sight
evolution on held-out tasks: 1.32 → 1.53, +0.20 [−0.12, +0.58], 9 wins / 19 ties / 9 losses.

Selected out-of-sight examples (more, including losses, in [GALLERY.md](GALLERY.md)):

**OE036: judged 1.0 → 5.0.** After the camera tilts up and returns, stock has replaced the man and dog; with the director they keep walking away. [video](gallery/outsight-OE036.mp4)

![OE036](gallery/outsight-OE036.jpg)

**OE030: judged 1.0 → 4.25.** Stock loses the ramp and the car; the director keeps the same car on the same ramp. [video](gallery/outsight-OE030.mp4)

![OE030](gallery/outsight-OE030.jpg)

{memv2_line}

## Demo script (2 minutes)

1. **The problem.** Play the left half of the hero video. The model looks away for 12 seconds, turns back, and the room it left is gone. It only remembers about 4.5 seconds.
2. **The fix.** Play the right half. We hand the model its own earlier frames of the place it is returning to. No training. The front door is back.
3. **It holds on scenes we never tuned on.** Show Figure 1. On 20 unseen scenes the judged score goes from {h12['stock'][0]:.2f} to {h12['memory'][0]:.2f} after 12 seconds away, {h12['wins']} wins to {h12['losses']} losses.
4. **It is cheap.** Show Figure 2. We keep the frame, not the model's internal state: 6 MB instead of 1.2 GB per frame, same recall.
5. **Where we stand, honestly.** We reproduced the published baseline twice (1.85 and 1.81 against 1.82). On PlayWorld's own tasks we are level with it, not above, because those tasks rarely look away for long. Long absences are where memory pays.

## Limits and what did not work

- **PlayWorld overall is unchanged.** First memory version 1.80 against stock 1.85; director 1.82 against 1.81. Neither is above Genie 3's 2.12, and the pre-registered criterion was not met (`reproduction/pass-criterion.md`).
- **PlayWorld's tasks rarely need memory.** Only 12% of geometry and out-of-sight chunks revisit a view the stock model had forgotten (`reproduction/full-run-memory-audit.md`).
- **The gain is in what the judge sees, not in our own similarity measure.** On the 20 unseen scenes, final-frame similarity to the start view is {h12['dino_stock']:.2f} against {h12['dino_memory']:.2f}. The judge credits consistent objects and layout more than an exact return.
- **The judge is noisy.** The same video scored three times varies by about 0.36 points, so single examples can mislead; that is why the table reports intervals.
- **Forward-and-back moves are not undone by the model**, and memory cannot fix that. A fine-tuned memory adapter (48 training pairs) did not help and hurt ordinary generation.

## What we would do next

- **A trained memory layer**: a small adapter that reads a compact summary of past views, in place of substituting state at all 40 layers.
- **Retrieval by what is visible**, as in VMem, so forward-and-back returns pick the right frames.
- **Longer tasks**: a benchmark split where the agent is away for 10 to 30 seconds, which is where worlds fall apart.

Sources for every number: [LEADERBOARD.md](../LEADERBOARD.md), [docs/METHOD.md](METHOD.md), `docs/demo/horizon-summary.json`,
and the files named above. Assets are built by `scripts/build_demo.py` and this page by `scripts/write_demo_page.py`.
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
    appendix += ["## Memory v2 on a slow 360° turn (GC007)", "",
                 "Both arms end on a similar plaza view, not the start view; the similarity scores are 0.57 (stock) and 0.62 (v2). "
                 "The point of this path is that v2 no longer scores below stock, as the first version did (0.49).", "",
                 "![slow 360](memv2-slow360-GC007.jpg)", ""]
    (DEMO / "APPENDIX.md").write_text("\n".join(appendix))


if __name__ == "__main__":
    main()
