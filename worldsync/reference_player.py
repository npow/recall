"""Reference-guided Claude controller with explicit per-model native timing."""
import base64
import hashlib
from io import BytesIO
import json

from PIL import Image

from worldsync.judge import parse_json

ACTIONS = {"W", "S", "A", "D", "LEFT", "RIGHT", "UP", "DOWN", "WAIT"}
OPERATIONS = {"KEEP", "STOP", "EXTEND", "CORRECT", "END"}


def decide(client, task, initial, latest, history, maximum=40, model="claude-haiku-4-5-20251001"):
    prompt = (
        "You are the Agent Player of an interactive world model. Pursue the goal using the human "
        "reference action plan as guidance and adapt to actual visual progress. The images show "
        "the initial world, then the latest generated observation. KEEP continues a reference "
        "action; STOP finishes the current action and selects the next appropriate action; "
        "EXTEND repeats an action when more progress is needed; CORRECT selects a corrective "
        "action; END terminates only when the goal is visually achieved. For all operations except "
        "END select exactly one action: W, S, A, D, LEFT, RIGHT, UP, DOWN, WAIT. W/S move "
        "forward/backward; A/D move sideways; arrows turn the camera; WAIT lets time pass with "
        "neutral controls. The first native chunk is 13 frames at 16 FPS; later chunks are "
        "16 frames (one second). Camera turns are commanded at 60 degrees per second. "
        "Scene identity drift is not evidence of returning to a location.\n"
        "Return JSON only: {\"operation\":\"KEEP|STOP|EXTEND|CORRECT|END\", "
        "\"action\":\"one action or null for END\", \"reason\":\"visual evidence\"}.\n"
        + json.dumps({"goal": task["prompt"], "reference_plan": task.get("action_sequence_steps") or task.get("action"),
                      "executed_actions": history, "max_interactions": maximum}, ensure_ascii=False)
    )
    content = [{"type": "text", "text": prompt}]
    for path in (initial, latest):
        image = Image.open(path).convert("RGB")
        image.thumbnail((1024, 1024))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
            "data": base64.b64encode(buffer.getvalue()).decode()}})
    response = client.messages.create(model=model, max_tokens=1500, messages=[{"role": "user", "content": content}])
    raw = "\n".join(block.text for block in response.content if block.type == "text")
    decision = parse_json(raw)
    if decision.get("operation") not in OPERATIONS or not isinstance(decision.get("reason"), str):
        raise ValueError("invalid player response")
    if decision["operation"] != "END" and decision.get("action") not in ACTIONS:
        raise ValueError("invalid player action")
    return decision, {"model": response.model, "response_id": response.id, "usage": response.usage.model_dump(),
        "raw_response": raw, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
