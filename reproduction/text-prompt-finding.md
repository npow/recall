# World-model text: task objective vs image caption

The released PlayWorld harness gives every world model the same initial frame and the task's objective:

* `external/PlayWorld/Agent_player/your-models/player.py`: `create_world` is documented as "Upload task.image_path and submit task.prompt".
* `external/PlayWorld/Agent_player/your-models/README.md`: "All adapters receive the same initial frame and long-horizon objective."
* `external/PlayWorld/Agent_player/SANA-WM/run_agent.py:860`: `prompt = task.get("prompt", "")` is passed to the model.

Our LingBot adapter (`worldsync/lingbot_rollout.py`) sent `task["image_caption"]`, the static description of the first frame,
so objectives such as "The woman keeps walking straight forward" never reached the model. For IF001 the caption is "standing in
front of a blue bus stop shelter", and the generated woman stood still while the camera pushed in (judged fail by both judges).
IF042's caption describes the man mid-jump and the video shows him floating, not walking.

`--text-prompt caption|prompt|both` (default `caption`, the unchanged baseline) now selects the text. Insight tasks use
"wait and observe" as their objective, so the caption is the more informative text there.

Status: not scored. The first `prompt` run on the 10 moving cases stopped after two cases when another process on the shared
GPU caused a CUDA out-of-memory error (GC007 1.0 vs 2.67 with caption and GC017 2.13 vs 2.0, two cases only). No benefit is claimed.
