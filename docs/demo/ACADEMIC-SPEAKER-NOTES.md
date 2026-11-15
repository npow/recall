# WorldSync — two-minute research presentation

Silent video for live presenting: **worldsync-academic-2min.mp4**. The clock below is the video timestamp. Approximately 300 spoken words; rehearse at a natural pace. The original demo cut remains available separately.

## 0:00–0:10 — Problem

World models generate the next frames in response to actions. But realistic frames are not enough. We ask: can the same world persist while an agent interacts with it?

## 0:10–0:22 — Task and benchmark

We start from frozen LingBot-World2 weights and evaluate with PlayWorld. Its four dimensions test geometry, instruction following, and process evolution both in sight and out of sight.

## 0:22–0:42 — Four failure videos

These are actual generated episodes. Top left, returning views drift. Top right, the character stops before completing the instruction. Bottom left, the piping tool changes identity instead of completing its process. Bottom right, a cup transforms, destroying the hidden process we wanted to observe.

## 0:42–1:02 — Method and architecture

We investigated two components. First, persistent camera state, precise action scheduling, and visual feedback keep control aligned with the task. Second, archived attention features can be retrieved when returning to an earlier view. Both use a frozen generator. We evaluated these separately; we have not demonstrated a combined improvement.

## 1:02–1:14 — Quantitative result

On three preselected instruction-following tasks, the corrected configuration improves from one point five six to two point four two. All three improve. This is an exploratory subset, not an overall benchmark win.

## 1:14–1:34 — Controller comparison

Here is the boardwalk task, with complete episodes accelerated three times. The legacy player ends early; that panel is explicitly held. Our configuration continues longer and earns a higher score, although the instruction is still only partly completed.

## 1:34–1:48 — Separate memory result

In a separate scripted revisit, archived visual memory improves endpoint similarity from point five seven four to point seven four six. This measures visual recovery, not a PlayWorld score.

## 1:48–2:00 — Limitations and future work

Objects still morph and processes reset. Next we will isolate prompt effects, combine control with memory, and test more seeds. Our result is promising components, not a solved world simulator.
