# WorldSync — two-minute research presentation

Silent video: **worldsync-academic-2min.mp4**. All clips play at normal speed. Five short side-by-side comparisons follow the four-quadrant overview. Approximately 275 spoken words; timings below follow the revised cut.

## 0:00–0:10 — Problem

World models generate frames in response to actions. But realistic frames are not enough. Can the same world persist while an agent interacts with it?

## 0:10–0:20 — Benchmark

We start from frozen LingBot-World2 weights. PlayWorld tests geometry, instruction following, and the evolution of processes both in sight and out of sight.

## 0:20–0:36 — Four failures

These are actual outputs at normal speed. Returning views drift, a walk stops early, a piping tool changes identity, and a cup transforms. Each failure breaks a different part of the world.

## 0:36–0:52 — Method and architecture

We investigated persistent camera control, retrieval of archived visual features, and a scene director that writes an event description for the generator. The weights stay frozen. These components were evaluated separately; we are not claiming a combined improvement.

## 0:52–1:02 — Scores

On three preselected instruction-following tasks, our configuration improves from one point five six to two point four two. This is exploratory, not an overall benchmark win.

## 1:02–1:14 — Boardwalk

The legacy player ends early, so its final frame is held. Our configuration continues longer and scores higher, although completing the instruction remains a problem.

## 1:14–1:26 — Pond

Here is a second task: walk left into the pond. The corrected configuration improves the full-episode score, but later appearance artifacts remain. These are short excerpts.

## 1:26–1:32 — Rowing

A separate scene director describes expected events. Here, rowing stays coherent.

## 1:32–1:38 — Pouring

Pouring improves too. These selected examples use one primary judgment per video.

## 1:38–1:48 — Process failure

Here our configuration does worse. The tool morphs instead of piping. Motion alone does not establish coherent progress, and geometric memory cannot solve every process failure.

## 1:48–2:00 — Next steps

Next: isolate prompt effects, combine control with retrieval, and track evolving processes. We need more seeds and held-out tests. These are promising components, with clear remaining failures.
