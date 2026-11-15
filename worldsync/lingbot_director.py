"""State-tracking text "director" for LingBot-World2 rollouts (EXPERIMENTAL, default off).

Stock rollouts give the world model one static text for the whole episode. On PlayWorld most
out-of-sight and in-sight evolution episodes then sit at the rubric floor because the scene freezes:
processes stop, and things that left the view do not progress. The director is a small LLM that
writes the world-model text so the world keeps evolving and entities keep their identity.

FIREWALL. The director sees ONLY these task fields: ``image_caption``, ``prompt`` (the task objective)
and ``action_sequence_steps`` (the human reference plan), plus run-time facts: elapsed commanded
seconds, the commanded camera action and the commanded camera state. It never receives the scored
questions, any ``*_policy`` field, rubric text, judge output, or generated images. ``director_inputs``
enforces this by whitelist and the unit tests check it.

Modes
* ``static``    one LLM call before generation: an enriched, dynamic description used for all chunks.
* ``perchunk``  the description is refreshed as time passes (every ``interval`` chunks or when the
                commanded action changes) so ongoing processes keep progressing.
* ``unseen``    ``perchunk`` plus view awareness: while the camera is turned away the director keeps
                advancing the off-screen process in its world state and, on return, describes the
                subjects in their advanced state.
"""
import json

MODES = ("off", "static", "perchunk", "unseen")
ALLOWED_TASK_FIELDS = ("image_caption", "prompt", "action_sequence_steps")
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
FIREWALL_NOTE = ("Director inputs: image_caption, prompt, action_sequence_steps, elapsed seconds, commanded action and "
                 "commanded camera state only. No questions, policies, rubric, judge output or images.")

TOOL = {"name": "write_world_text", "description": "Write the world-model text for the next seconds of video.",
        "input_schema": {"type": "object", "additionalProperties": False,
                         "properties": {"world_state": {"type": "string"}, "prompt": {"type": "string"}},
                         "required": ["world_state", "prompt"]}}

BASE_RULES = (
    "You are the director of an interactive generative video world model. Every second it renders the next "
    "chunk of video from (a) a text description you write, (b) a camera command chosen by a separate player and "
    "(c) its recent frames. Left alone, its scenes tend to freeze: people stand still, ongoing processes stop, "
    "and anything that leaves the view does not progress. Your job is to keep the world alive and consistent.\n"
    "Rules for the text you write (field `prompt`):\n"
    "1. Start from the original scene description and keep every entity's identity and appearance exactly: same "
    "people, clothing, colours, objects, counts and layout. Never add new entities, never cut to another scene.\n"
    "2. Describe what is happening as continuous visible motion in the present progressive, with concrete "
    "physical detail (who or what moves, how, in which direction). Ongoing processes advance naturally with the "
    "elapsed time; nothing teleports and nothing resets.\n"
    "3. If the task objective says a subject does something (for example keeps walking forward), describe that "
    "subject doing exactly that, steadily and physically plausibly, including contact with whatever is in its way.\n"
    "4. Do not give camera instructions (the camera is controlled separately), no style or quality words, no "
    "mention of prompts, tasks, players, time stamps or these rules.\n"
    "5. One paragraph of plain descriptive English, at most 90 words.\n"
    "Field `world_state` is your private memory (at most 70 words): the entities, the process under way, how far "
    "it has progressed, and what is currently out of view. Keep it consistent with your previous world_state."
)
UNSEEN_RULES = (
    "\nView awareness: you are told how far the camera has turned or moved away from the original view. While the "
    "original subjects are out of frame, keep advancing their process in `world_state` (it continues unseen) and "
    "let `prompt` describe what is plausibly visible in the current direction of the same place, still alive and "
    "consistent with the setting. When the camera comes back, `prompt` must describe the same subjects in their "
    "advanced state (more of the work done, the moving thing further along), with identical identity and appearance."
)


def director_inputs(task):
    """The only task fields the director may see (firewall)."""
    return {field: task.get(field) for field in ALLOWED_TASK_FIELDS}


def view_note(commanded_state):
    """Plain-language camera state from commanded poses only (no images)."""
    if not commanded_state:
        return "camera at the original view"
    heading = float(commanded_state.get("heading_degrees", 0.0))
    position = commanded_state.get("position_normalized_units") or [0.0, 0.0, 0.0]
    distance = sum(float(value) ** 2 for value in position) ** 0.5
    wrapped = (heading + 180.0) % 360.0 - 180.0
    if abs(wrapped) < 25:
        facing = "facing the original direction"
    else:
        facing = f"turned {abs(wrapped):.0f} degrees to the {'right' if wrapped > 0 else 'left'} of the original direction"
    away = "the original subjects are in frame" if abs(wrapped) < 40 else "the original subjects are out of frame"
    moved = "at the starting position" if distance < 1.5 else f"moved about {distance:.0f} steps from the starting position"
    return f"camera {facing}, {moved}; {away}"


class Director:
    def __init__(self, client, task, mode, model=DEFAULT_MODEL, interval=4):
        if mode not in MODES or mode == "off":
            raise ValueError(f"director mode must be one of {MODES[1:]}")
        self.client, self.mode, self.model, self.interval = client, mode, model, max(1, int(interval))
        self.inputs = director_inputs(task)
        self.world_state = ""
        self.text = None
        self.last_action = None
        self.last_call_chunk = None
        self.calls = []

    def system(self):
        return BASE_RULES + (UNSEEN_RULES if self.mode == "unseen" else "")

    def _ask(self, situation, chunk):
        payload = {"scene_description": self.inputs["image_caption"], "task_objective": self.inputs["prompt"],
                   "reference_plan": self.inputs["action_sequence_steps"], "previous_world_state": self.world_state,
                   "previous_text": self.text, **situation}
        response = self.client.messages.create(
            model=self.model, max_tokens=600, system=self.system(), tools=[TOOL],
            tool_choice={"type": "tool", "name": "write_world_text"},
            messages=[{"role": "user", "content": json.dumps(payload)}])
        block = next((item for item in response.content if getattr(item, "type", None) == "tool_use"), None)
        if block is None or not str(block.input.get("prompt", "")).strip():
            raise ValueError("director returned no text")
        text = " ".join(str(block.input["prompt"]).split())
        self.world_state = " ".join(str(block.input.get("world_state", "")).split())
        self.calls.append({"chunk": chunk, **situation, "world_state": self.world_state, "prompt": text})
        self.last_call_chunk = chunk
        return text

    def initial(self):
        """Text for the start of the episode (the only call in ``static`` mode)."""
        situation = {"elapsed_seconds": 0.0, "situation": "start of the episode; the camera is at the original view"}
        try:
            self.text = self._ask(situation, -1)
        except Exception as error:  # never let the director kill an episode: fall back to the caption
            self.calls.append({"chunk": -1, "error": f"{type(error).__name__}: {error}"[:300]})
            self.text = self.inputs["image_caption"]
        return self.text

    def for_chunk(self, index, action, commanded_state, elapsed_seconds):
        """Text for chunk ``index``; returns (text, changed)."""
        if self.text is None:
            self.initial()
        if self.mode == "static":
            return self.text, False
        due = self.last_call_chunk is None or self.last_call_chunk < 0 or index - self.last_call_chunk >= self.interval
        action_changed = self.last_action is not None and action != self.last_action
        self.last_action = action
        if index == 0 or not (due or action_changed):
            return self.text, False
        situation = {"elapsed_seconds": round(float(elapsed_seconds), 2), "camera_action_now": action}
        if self.mode == "unseen":
            situation["camera_state"] = view_note(commanded_state)
        previous = self.text
        try:
            self.text = self._ask(situation, index)
        except Exception as error:
            self.calls.append({"chunk": index, "error": f"{type(error).__name__}: {error}"[:300]})
            return self.text, False
        return self.text, self.text != previous

    def metadata(self):
        return {"mode": self.mode, "model": self.model, "interval_chunks": self.interval, "firewall": FIREWALL_NOTE,
                "allowed_task_fields": list(ALLOWED_TASK_FIELDS), "calls": self.calls,
                "llm_calls": sum(1 for call in self.calls if "error" not in call),
                "errors": sum(1 for call in self.calls if "error" in call)}


class PresetDirector:
    """Same interface as ``Director`` but the texts come from a committed JSON file (no LLM call).

    File layout: {"cases": {"<split>:<TASK>": {"static": text, "scheduled": [[first_chunk, text], ...]}}}.
    ``static`` mode uses the single text; ``perchunk``/``unseen`` use the scheduled segments (falling back to
    the static text when a case has no schedule).
    """

    def __init__(self, path, split, task_id, mode):
        if mode not in MODES or mode == "off":
            raise ValueError(f"director mode must be one of {MODES[1:]}")
        with open(path) as handle:
            entry = json.load(handle)["cases"].get(f"{split}:{task_id}")
        if not entry:
            raise ValueError(f"no preset director text for {split}:{task_id}")
        self.mode, self.path = mode, str(path)
        schedule = entry.get("scheduled") if mode != "static" else None
        self.segments = sorted((int(start), " ".join(text.split())) for start, text in (schedule or [[0, entry["static"]]]))
        if self.segments[0][0] != 0:
            raise ValueError("the first scheduled segment must start at chunk 0")
        self.text = None
        self.calls = []

    def _segment(self, index):
        return [text for start, text in self.segments if start <= index][-1]

    def initial(self):
        self.text = self._segment(0)
        self.calls.append({"chunk": -1, "prompt": self.text})
        return self.text

    def for_chunk(self, index, action, commanded_state, elapsed_seconds):
        if self.text is None:
            self.initial()
        text = self._segment(index)
        changed = text != self.text
        if changed:
            self.text = text
            self.calls.append({"chunk": index, "camera_action_now": action, "prompt": text})
        return self.text, changed

    def metadata(self):
        return {"mode": self.mode, "model": "preset text file (no LLM call)", "text_file": self.path,
                "segments": len(self.segments), "firewall": FIREWALL_NOTE, "calls": self.calls, "llm_calls": 0, "errors": 0}
