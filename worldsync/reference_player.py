"""Reference-guided Claude controller with explicit per-model native timing."""
import base64
import hashlib
from io import BytesIO
import json
import re

from PIL import Image

ACTIONS = {"W", "S", "A", "D", "LEFT", "RIGHT", "UP", "DOWN", "WAIT"}
OPERATIONS = {"KEEP", "STOP", "EXTEND", "CORRECT", "END"}
CONTROL_TOOL = {"name": "choose_action", "description": "Select one camera action toward the goal.",
    "strict": True, "input_schema": {"type": "object", "additionalProperties": False,
        "properties": {"operation": {"type": "string", "enum": sorted(OPERATIONS)},
                       "action": {"type": "string", "enum": sorted(ACTIONS) + ["NONE"]},
                       "reason": {"type": "string"}}, "required": ["operation", "action", "reason"]}}


class PlayerResponseError(ValueError):
    def __init__(self, message, metadata):
        super().__init__(message)
        self.metadata = metadata


def held_seconds(history):
    """Seconds the most recent action has been continuously commanded.

    The first native chunk is 13 frames (0.8125 s); later chunks are 1 s.
    """
    if not history or not history[-1].get("action"):
        return 0.0
    count = 0
    for item in reversed(history):
        if item.get("action") != history[-1]["action"]:
            break
        count += 1
    return count - (0.1875 if count == len(history) else 0)


PLAN_STEP = re.compile(r"(hold|wait)\((?:([A-Za-z]+),)?(\d+)ms\)")
PLAN_ACTIONS = {"w": "W", "s": "S", "a": "A", "d": "D", "left": "LEFT", "right": "RIGHT", "up": "UP", "down": "DOWN"}


def plan_steps(task):
    """Reference plan as [{"action","seconds"}] parsed from hold(KEY,ms)/wait(ms)."""
    steps = []
    for text in task.get("action_sequence_steps") or []:
        match = PLAN_STEP.fullmatch(text.strip())
        if match:
            kind, key, ms = match.groups()
            action = "WAIT" if kind == "wait" else PLAN_ACTIONS.get((key or "").lower())
            if action:
                steps.append({"action": action, "seconds": int(ms) / 1000})
    return steps


def elapsed_seconds(history):
    """Commanded world time: first native chunk 0.8125 s, then 1 s per chunk."""
    return max(0.0, len(history) - 0.1875) if history else 0.0


def decide(client, task, initial, latest, history, maximum=40, model="claude-haiku-4-5-20251001", commanded_state=None):
    steps = plan_steps(task)
    plan_total = round(sum(step["seconds"] for step in steps), 3)
    elapsed = round(elapsed_seconds(history), 3)
    rules = (
        "Follow the reference plan's actions in order and hold each for about its planned seconds "
        "(within one chunk); compare current_action_held_seconds and elapsed_commanded_seconds with "
        "plan_steps and plan_total_seconds. Do not invent extra legs or turns that neither the plan nor "
        "the goal needs. Do not END before the plan's total time is consumed unless the goal is already "
        "unambiguously complete in the images; for revisit goals compare the latest image with the "
        "initial image before END. A subject and background that scale together mean camera motion, "
        "not the subject walking. "
    )

    def ask(note=None):
        prompt = (
            "You are the Agent Player of an interactive world model. Pursue the goal using the human "
            "reference action plan as guidance and adapt to actual visual progress. The goal takes "
            "priority over the reference plan only when the plan literally conflicts with it. The images show "
            "the initial world, then the latest generated observation. KEEP continues a reference "
            "action; STOP finishes the current action and selects the next appropriate action; "
            "EXTEND repeats an action when more progress is needed; CORRECT selects a corrective "
            "action; END terminates only when the goal is visually achieved. For all operations except "
            "END select exactly one action: W, S, A, D, LEFT, RIGHT, UP, DOWN, WAIT. W/S move "
            "forward/backward without turning; A/D move sideways; arrows turn the camera; WAIT lets time pass with "
            "neutral controls. The first native chunk is 13 frames at 16 FPS; later chunks are "
            "16 frames (one second). Walking backward (S) does not turn around. "
            "Use intended odometry to track commands and avoid endless repeats, but verify realized "
            "motion and completion visually; odometry alone cannot prove success. "
            "Scene identity drift is not evidence of returning to a location. " + rules + "\n"
            "Call choose_action exactly once. Use action=NONE only for END; otherwise choose "
            "one permitted action and give visual evidence in reason.\n"
            + json.dumps({"goal": task["prompt"], "reference_plan": task.get("action_sequence_steps") or task.get("action"),
                          "plan_steps": steps, "plan_total_seconds": plan_total, "elapsed_commanded_seconds": elapsed,
                          "executed_actions": history, "intended_odometry": commanded_state,
                          "current_action": history[-1].get("action") if history else None,
                          "current_action_held_seconds": held_seconds(history),
                          "max_interactions": maximum, **({"controller_note": note} if note else {})}, ensure_ascii=False)
        )
        content = [{"type": "text", "text": prompt}]
        for path in (initial, latest):
            image = Image.open(path).convert("RGB")
            image.thumbnail((1024, 1024))
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                "data": base64.b64encode(buffer.getvalue()).decode()}})
        response = client.messages.create(model=model, max_tokens=1500,
            tools=[CONTROL_TOOL], tool_choice={"type": "tool", "name": "choose_action", "disable_parallel_tool_use": True},
            messages=[{"role": "user", "content": content}])
        calls = [block.input for block in response.content if block.type == "tool_use" and block.name == "choose_action"]
        metadata = {"model": response.model, "response_id": response.id, "usage": response.usage.model_dump(),
            "raw_response": json.dumps(calls), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
        if len(calls) != 1:
            metadata["raw_response"] = str(response.content)
            raise PlayerResponseError("expected exactly one player control call", metadata)
        decision = calls[0]
        if decision.get("operation") == "END":
            decision = {**decision, "action": None}
        if decision.get("operation") not in OPERATIONS or not isinstance(decision.get("reason"), str):
            raise PlayerResponseError("invalid player response", metadata)
        if decision["operation"] != "END" and decision.get("action") not in ACTIONS:
            raise PlayerResponseError("invalid player action", metadata)
        return decision, metadata

    decision, metadata = ask()
    if decision["operation"] == "END" and plan_total and elapsed < plan_total - 1.0:
        rejected = decision["reason"]
        decision, second = ask(f"END was proposed at {elapsed}s but the reference plan lasts {plan_total}s. "
            "Continue the plan unless the images already show the goal unambiguously achieved; if you still END, "
            "cite the specific visual evidence.")
        metadata = {**second, "first_attempt_rejected_end": rejected, "first_attempt_response_id": metadata["response_id"],
            "first_attempt_usage": metadata["usage"]}
    return decision, metadata
