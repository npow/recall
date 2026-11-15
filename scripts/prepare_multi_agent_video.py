"""Prepare two character controllers' basket handoff for a LingBot clip.

This is an open-loop character-control experiment, not two first-person cameras.
The recorded joint actions become scheduled conditioning of one causal rollout.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCENE = ("Two women are sitting on the grassy riverbank looking across the river. "
         "The woman on the left wears a purple floral kimono with a pale green bow. "
         "The woman on the right wears an orange floral kimono with a silver bow. "
         "There is exactly one brown woven basket. Ducks swim on the calm river, "
         "with the same houses, stone embankment and green trees across the water. ")
TOOL = {"name": "act", "description": "Choose your character's next action and a message to the other agent.",
        "input_schema": {"type": "object", "additionalProperties": False,
                         "properties": {"action": {"type": "string", "enum": ["wait", "pick_up", "offer", "accept"]},
                                        "message": {"type": "string"}}, "required": ["action", "message"]}}


def prepare(output, client=None):
    output.mkdir(parents=True, exist_ok=True)
    state = {"revision": 0, "basket": "ground_right", "offer_to": None}
    histories = {actor: [] for actor in ("orange", "purple")}
    trace, scheduled = [], []
    motion = {
        ("orange", "pick_up"): "The orange-kimono woman is leaning right, reaching for the single basket and lifting it onto her lap. The purple-kimono woman is turning toward her and waiting.",
        ("orange", "offer"): "The orange-kimono woman is turning left and slowly extending the brown basket toward the purple-kimono woman. The purple-kimono woman is turning right and reaching both hands toward the offered basket.",
        ("purple", "accept"): "The purple-kimono woman is receiving the single brown basket from the orange-kimono woman's hands and drawing it onto her lap. The orange-kimono woman is releasing the basket and withdrawing her empty hands.",
    }
    phases = [("orange", "pick_up"), ("orange", "offer"), ("purple", "accept")]
    for phase, (actor, required) in enumerate(phases):
        other = "purple" if actor == "orange" else "orange"
        payload = {"shared_state": state.copy(), "your_identity": actor,
                   "goal": "Transfer the one basket from the ground on the right to the purple woman's lap. Coordinate through messages. You can act only as your own character.",
                   "latest_message": trace[-1]["message"] if trace else "Orange, please bring me the basket.",
                   "valid_progress_action": required}
        if client is None:
            decision = {"action": required, "message": {"pick_up": "I have the basket and will pass it to you.",
                        "offer": "The basket is in my hands, offered to you. Please take it.",
                        "accept": "I have received the basket. You can let go."}[required]}
        else:
            response = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=300,
                system="You control one character in a shared world. Use the current committed state and your partner's message. Never invent a completed action. Choose the valid progress action when its preconditions hold.",
                tools=[TOOL], tool_choice={"type": "tool", "name": "act"},
                messages=histories[actor]+[{"role": "user", "content": json.dumps(payload)}])
            decision = next(block.input for block in response.content if block.type == "tool_use")
        if decision["action"] != required:
            raise ValueError(f"{actor} chose {decision['action']} instead of the valid action {required}")
        if required == "pick_up" and state["basket"] == "ground_right":
            state["basket"] = "orange"
        elif required == "offer" and state["basket"] == "orange":
            state["offer_to"] = other
        elif required == "accept" and state["offer_to"] == actor and state["basket"] == "orange":
            state.update(basket="purple", offer_to=None)
        else:
            raise ValueError("invalid shared-state transition")
        state["revision"] += 1
        trace.append({"actor": actor, "input": payload, **decision, "committed_state": state.copy()})
        histories[actor].extend([{"role": "user", "content": json.dumps(payload)},
                                 {"role": "assistant", "content": json.dumps(decision)}])
        scheduled.append([phase*6, SCENE+motion[actor, required]])
    scheduled.append([18, SCENE+"The purple-kimono woman is holding the one brown basket on her lap. The orange-kimono woman has empty hands. They are turning back toward the river, sitting side by side as the ducks continue swimming."])
    plan = {"scope": "two character controllers, shared symbolic state, one shared camera; open-loop rendering",
            "controller": "scripted" if client is None else "LLM",
            "ipc": "explicit local messages; past.dev not configured", "actions": trace, "final_state": state,
            "model": None if client is None else "claude-haiku-4-5-20251001", "chunks": 24}
    (output / "agent-plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    (output / "director.json").write_text(json.dumps({"cases": {"insight:OE025": {
        "static": scheduled[0][1], "scheduled": scheduled}}}, indent=2)+"\n")
    return plan


if __name__ == "__main__":
    from worldsync.credentials import load_local_credentials
    load_local_credentials()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/multi-agent-video-plan")
    parser.add_argument("--scripted", action="store_true", help="explicit scripted controllers; no agent API calls")
    args = parser.parse_args()
    client = None
    if not args.scripted:
        from anthropic import Anthropic
        client = Anthropic(timeout=90, max_retries=1)
    plan = prepare(args.output, client)
    print(json.dumps({"final_state": plan["final_state"], "actions": len(plan["actions"]), "output": str(args.output)}))
