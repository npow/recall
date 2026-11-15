# What caps out-of-sight (OE) and insight (IE) scores

Read-only analysis of existing Gemini (Vertex, `gemini-3.1-pro-preview`) judgements. No new generation.
Scores are local-protocol numbers, not comparable with the paper's 1.82 / 2.12.

## Data actually available

* Powered test, seeds 42 and 43 (worktree `agent-a5f22f4098cb528aa`, `outputs/powered-gemini/episodes/episodes/*.json`):
  **42 OE episodes = 7 outsight tasks (OE026, OE029, OE044, OE055, OE056, OE057, OE069) x 3 arms (stock, budget memory, 46/6) x 2 seeds.**
* **No insight episode has ever been judged.** All 42 powered OE episodes are from the `outsight` split, and the 75 episodes in
  `reproduction/gemini-rescore/` are 51 GC and 24 outsight. The insight dimension is analysed from its objectives only (section 4).
* Episode JSONs keep the full Gemini evidence (`evidence_zh` per category, `oe_dynamicity_check`, gate reasons). Player actions are in each
  episode's `run.json`.

## 1. Outcome distribution (42 outsight episodes)

Gate verdict: pass 24, partial 7, **fail 11** (a gate fail scores exactly 1.0).

Each arm has 14 episodes. Columns: gate fail (scored 1.0), then the primary cause among the scored episodes: (a) hidden process frozen or wrong,
(b) identity replaced or object vanished on return, (c) physics or collision, (d) trajectory not completed, then clean (score 5.0).

| Arm | gate fail | (a) | (b) | (c) | (d) | clean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stock | 5 | 3 | 4 | 1 | 1 | 0 |
| budget memory | 2 | 3 | 8 | 1 | 0 | 0 |
| 46/6 | 4 | 2 | 3 | 2 | 1 | 2 |

Totals: 11 gate fails; 31 scored = (a) 8, (b) 15, (c) 4, (d) 2, clean 2. My labels, one primary cause per episode from the judge's category evidence (reading the evidence text, not a script).
The only two episodes scoring 5.0 are **OE029 and OE069 on 46/6, seed 43**. 46/6 with seed 42 on the same tasks scores 3.0 and 1.0.
So an arm can succeed on a case, but not reliably.

## 2. The causes, with episode ids

**(b) Identity replacement or an object vanishing after the camera comes back: 15 of 31 scored episodes. The largest bucket.**
It happens within a window the stock cache nominally covers, because the out-of-view interval is only about 3.6 s (a 90 deg turn and back).

* OE026 (all 5 scored episodes): after the turn back, "the woman's face changed completely, the girl's white clothes became striped, the
  rabbit went from lop-eared to upright" (stock s42, 1.75). Even 46/6 s43 (3.0): "the black-and-white rabbit was replaced by a pure white one".
* OE055 stock s43: the tall graduated beaker was "replaced by a short glass jar", so the diffusion process could not continue.
* OE056 stock s43: the red dye vanished after the beaker re-entered, and a gloved hand appeared from nowhere. 46/6 s43: the red liquid vanished and the
  dropper's liquid flipped from red to blue.
* OE044 (budget s42, s43, stock s43): the tomato slices or the knife vanish, or the scene resets, on return. The fourth scored OE044 episode (46/6 s42)
  is counted under (d); its slices also vanish on the second return.
* OE069 budget s42: a clone of the mother and daughter appears behind the camera during a 180 deg turn ("cloned in the flat area behind").
  The process itself was fine (progression 5/5) and the score was still 3.0 because identity was 1/5.

**(a) Hidden process frozen or wrong while out of view: 8 of 31.** No memory or retrieval method addresses this.

* OE029 stock s42: "the train stopped moving after the camera turned, and the camera ended up inside the train". Stock s43: the locomotive
  reverses and a new carriage appears. Budget s43: the locomotive teleports backwards.
* OE055 budget s43, 46/6 s43: "the diffusion process is completely frozen".
* OE069 46/6 s42, stock s42: the pair on the escalator does not move down while out of view (position rises, or does not change).
* OE056 budget s43: mixing stalls and the red stream becomes a solid rod.

**(c) Physics and collision: 4 of 31.** OE029 budget s42 and 46/6 s42 (both 3.0): the camera clips through the train; OE055 stock s42 and
46/6 s42: the dropper follows the camera and drips onto the table.

**(d) Trajectory not completed, gate partial: 2 of 31.** OE044 46/6 s42 (the world executed only the first turn); OE069 stock s43 (kept panning and never
returned). These two partials are control-realization failures by the judge's account. I did not check them visually.

**Clean: 2 of 31** (46/6 s43 on OE029 and OE069). In both, the process continued (train keeps moving; escalator keeps carrying) and identity
held. That is the target behaviour, and nothing in the stock or budget arms reaches it.

## 3. The gate: a probable judge-noise floor on OE

11 of 42 OE episodes (26%) are gate fails, and every one scores 1.0. The judge's reasons are of two kinds:

* camera "moved right instead of left" or "did not turn back": OE057 x5 (stock s42, s43, budget s43, 46/6 s42, s43) and OE026 stock s43;
* camera "completely static": OE056 x3 (stock s42, budget s42, 46/6 s42) and OE044 x2 (stock s42, 46/6 s43).

In every one of these the Haiku player **commanded the right thing** (e.g. OE057: LEFT, LEFT, RIGHT, RIGHT; OE056: A x5, D x5).
I looked at 12-frame contact sheets of three of the eleven:

* **OE057 stock s42** (judge: "panned right, never returned"): the blue-shirted baker leaves the frame to the right while a black-apron person and
  the table enter, then the blue shirt returns. A leftward turn moves content to the right. The camera did turn left and back.
* **OE056 stock s42** (judge: "completely static"): the beaker moves from left-centre to right-centre and back across the frames. The camera moved.
* **OE026 stock s43** (judge: "turned right, not left"): the content shifts and returns; the person on return is a different (long-haired) woman.
  This looks like a correct turn plus an identity failure (cause b), not a wrong-direction turn.

So in the three I checked the camera did what was commanded and the gate verdict looks wrong. **Eight of the eleven are unchecked.** If the pattern holds:

* a quarter of OE episodes are floored at 1.0 for reasons that have nothing to do with the world model;
* it applies to all arms (gate fails: stock 5, budget 2, 46/6 4), so it adds noise and compresses every OE delta;
* the powered test's "+0.54 OE for budget memory" comes mostly from budget having 2 gate fails against stock's 5, and budget has **no** clean episode.
  I would not read that gain as a memory effect.

The paper's hosted LingBot also scores only 1.16 on OE, which fits the same floor, but I cannot prove that.

## 4. Insight and IF objectives

Insight (30 tasks): **27 of 30 are `wait*N` only ("wait and observe") with a static camera; 1 is "slowly move the camera to the right".**
The released harness has an `is_oe_insight_wait_only` special case. There is no camera path, no revisit, and nothing for a player to do.
The score is the world model's own dynamics from the starting image and its text. Retrieval, cache size and the player cannot affect it.
Examples: OE001 pedestrian with stroller crossing; OE010 pedestrians on a crosswalk; OE013 knife cutting pineapple; OE040 cat grooming;
OE065 cashier serving; OE072 hands tearing flatbread; OE004 toddler; OE022 cattle; OE007 gameplay shot.
Camera-only controls cannot attempt anything beyond WAIT. Per-chunk text could ("the woman keeps pushing the stroller across"), but the objective is
literally "wait and observe", so any process text would have to come from the image caption.

IF (50 tasks, 47 third person): 27 are a bare forward walk `w*N` ("The woman walks forward along the dirt path toward the sea") and
**8 need `interact()` actions** (IF016, IF027, IF030, IF033 robotic arm grips a drawer handle and pulls, IF034, IF037, IF038 walk along a wall then jump, IF041).

| Task | Requires | Camera poses plus per-chunk text can attempt it? |
| --- | --- | --- |
| IF001, IF044 (woman keeps walking, even through obstacles), IF012, IF017, IF023, IF007 (walks down stairs) | a third-person subject moving forward | W moves the camera, not the subject. The subject only walks if the **text** says so. Yes via text, partially |
| IF049 (girl turns right, then walks forward) | turn plus walk | camera turn yes; the subject's turn and walk by text |
| IF028 (horse gallops and crashes into an archway) | alternating w/a, a violent collision | text only; collision physics is a known weak spot (OE029 clipping) |
| IF033 (robotic arm grips and pulls a drawer) | `interact(grip)`, `interact(pull)` | **no camera analogue**; text only |
| IF038 (walk on a wall 10 m, then jump down) | `w*10` then `interact(jump)` | walk by text; the jump by text only |

## 5. A protocol difference that matters for IF

The released PlayWorld harness passes **`task["prompt"]` (the objective) as the world model's text** (`Agent_player/SANA-WM/run_agent.py`, `prompt = task.get("prompt", "")`
then `engine.generate(img_path, prompt, ...)`). Our rollout defaults to `--text-prompt caption` (the static image caption), and the powered
episodes have no `text_prompt_mode` recorded, so they used the caption. `--text-prompt prompt|both` exist but are gated behind the hillclimb unlock.
For IF that is probably decisive (the caption never says the subject walks; the objective does). For OE and insight it changes little
(the OE objective is camera instructions, the insight objective is "wait and observe").
This is a protocol deviation in our baseline, not a tuning lever, and it must be settled before IF and insight numbers mean anything.

## 6. Where the headroom is

| Lever | Evidence | Verdict |
| --- | --- | --- |
| Cache and retrieval memory | cause (b) is 15 of 31 scored OE episodes, and clean successes occur only with 46/6 (2 of 14, both seed 43). Budget memory has 8 identity failures and no clean episode | **Largest real headroom, but not shown to be reliable.** Seed variance is as large as the arm effect |
| State tracker or director (per-chunk text from caption and elapsed time, never rubric text) | cause (a) is 8 of 31: frozen or wrong process. The 5.0 episodes are the ones where the process continued | Plausible, **untested**; only helps (a) and some (b) |
| Player | the player commanded the correct action in all 11 gate fails; 2 partials (d) are control-realization | **Not the cap.** Early END is handled by `--end-guard` |
| Judge and gate | 26% of OE episodes floored at 1.0; 3 of 3 checked look like judge errors | **Largest single suppressor of OE scores**, but not a lever we control |
| IF text prompt (`--text-prompt prompt`) | caption-only runs cannot make the subject walk | **Must test before reading any IF number** |
| Insight | 27 of 30 are wait-only | nothing in memory or the player applies; only the model's own dynamics and the text channel |

## 7. Caveats

* Seven outsight tasks, two seeds, one Gemini sample per episode: 42 episodes in all, no insight episode.
* My cause labels are my reading of the judge's evidence, not an independent measurement. Only three videos were checked.
* The 5.0 outliers are one seed on two tasks. Do not read them as a method win.
* The evidence text is the judge's; it can be wrong about identity as well as direction.

Sources: `outputs/powered-gemini/episodes/episodes/*.json` and `outputs/powered-episodes/*/run.json` in worktree `agent-a5f22f4098cb528aa`;
`/home/npow/code/iteratehack/external/PlayWorld/data/{outsight,insight,if}/data.json`; `Agent_player/SANA-WM/run_agent.py`; `worldsync/lingbot_rollout.py` (`--text-prompt`).
