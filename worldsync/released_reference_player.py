"""Explicit HY WorldPlay reference-controller transfer, not paper LingBot parity.

Pinned released prompt, parser, action parsing and observation preprocessing run
unchanged. Browser wall-clock scheduling is adapted to discrete generated chunks;
Anthropic's SDK transports the same text/image request without a custom tool.
"""

import argparse
import ast
import base64
import copy
import hashlib
import json
import re
import time
import weakref
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from worldsync.reference_player import PlayerResponseError

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external/PlayWorld/Agent_player/hyworld2/player.py"
SOURCE_SHA256 = "d7a90541a19c27cf4c7ed298510fb1585974c85bdd90ea39547150093b287291"
MODEL = "claude-haiku-4-5-20251001"
PROFILE = "released-hy-reference-guided-transfer-v1"
FUNCTIONS = {
    "request_claude_hold_decision",
    "extract_chat_text",
    "extract_json_object",
    "parse_actions",
    "claude_viewer_screenshot",
}
CONSTANTS = {
    "KEY_MAP",
    "ARROW_KEYS",
    "CLAUDE_ADAPTIVE_KEY_MAP",
    "ACTION_UNIT_SECONDS",
    "ROTATION_EXTRA_SECONDS",
    "LS_ROTATION_HOLD_SECONDS",
}
KEYS = {
    "KeyW": "W",
    "KeyA": "A",
    "KeyS": "S",
    "KeyD": "D",
    "ArrowLeft": "LEFT",
    "ArrowRight": "RIGHT",
    "ArrowUp": "UP",
    "ArrowDown": "DOWN",
}
DIFFERENCES = [
    "No public LingBot player exists; this transfers HY-specific reference-guided behavior, not verified paper LingBot behavior.",
    "Native generation pauses during SDK calls; released browser controls remain held while the request runs.",
    "Durations use generated time (first chunk812.5ms, later1000ms), rounded at chunk boundaries instead of browser wall time.",
    "First live check observes the first generated chunk instead of starting immediately after browser keydown.",
    "Released HY action-field priority,1.5s action units and1.5s arrow padding differ from derived action_sequence_steps and cannot be assumed to match hosted LingBot timing.",
    "Released prompt uses the latest screenshot only, not an initial/current image pair or arbitrary history.",
    "Decoded native frames replace browser screenshots; the released JPEG65 then640x360/JPEG45 preprocessing is retained.",
    "Anthropic SDK replaces the released OpenAI-compatible HTTP proxy; malformed/API failures abort instead of silently continuing.",
    "Neutral observation tail ends at the native chunk budget, not HY's natural60s world endpoint; END does not assert visual success.",
    "A40interaction paper limit is not verified equivalent to40one-second native chunks; some released HY reference plans exceed this chunk budget.",
    "Only native camera controls are supported; avatar interactions and simultaneous keys are rejected, not synthesized.",
]


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _load_source(path):
    raw = Path(path).read_bytes()
    if _sha(raw) != SOURCE_SHA256:
        raise ValueError("Released HY player source differs from the audited snapshot")
    text = raw.decode()
    tree = ast.parse(text)
    namespace = {
        "argparse": argparse,
        "base64": base64,
        "time": time,
        "json": json,
        "re": re,
        "Image": Image,
        "BytesIO": BytesIO,
    }
    nodes, hashes = [], {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in CONSTANTS:
                    namespace[target.id] = ast.literal_eval(node.value)
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            nodes.append(node)
            hashes[node.name] = _sha(ast.get_source_segment(text, node).encode())
    if set(hashes) != FUNCTIONS or not CONSTANTS <= namespace.keys():
        raise ValueError("Released HY controller definitions are incomplete")
    # Preserve annotations as strings without importing Playwright or executing
    # the browser runner, environment setup, API clients or module-level code.
    import __future__

    compiled = compile(
        ast.Module(body=nodes, type_ignores=[]),
        str(path),
        "exec",
        flags=__future__.annotations.compiler_flag,
    )
    exec(compiled, namespace)  # noqa: S102 -- SHA-pinned pure function AST, no runner/module execution
    return namespace, hashes


class _ImageSink:
    @property
    def parent(self):
        return self

    def mkdir(self, **kwargs):
        pass

    def write_bytes(self, value):
        pass


class ReleasedReferenceController:
    """One instance per episode; its decide method matches the legacy callback."""

    def __init__(self, source_path=SOURCE):
        self.namespace, self.function_hashes = _load_source(source_path)
        self.namespace["post_claude"] = self._post
        self.actions = []
        self.state = None
        self.issued = []
        self.signature = None
        self.client = None
        self.transport = None

    def metadata(self):
        return {
            "profile": PROFILE,
            "model": MODEL,
            "released_source_sha256": SOURCE_SHA256,
            "released_function_sha256": self.function_hashes,
            "exact_paper_lingbot_controller": False,
            "reference_guided": True,
            "scheduler_policy": {
                "reference_priority": "released parse_actions: action field before action_sequence_steps",
                "action_unit_seconds": 1.5,
                "arrow_padding_ms": 1500,
                "minimum_hold_ms": 1000,
                "minimum_release_ms": 1000,
                "maximum_checks_per_hold": 3,
                "minimum_check_spacing_ms": 1500,
                "maximum_extension_per_check_ms": 5000,
                "maximum_total_extension_ms": 10000,
                "maximum_unique_post_hold_arrow_corrections": 2,
                "correction_hold_ms": 1000,
                "live_checks_on_rotation_holds": False,
            },
            "remaining_differences": DIFFERENCES,
        }

    def validate_task(self, task):
        actions = self.namespace["parse_actions"](task)
        unsupported = [
            action
            for action in actions
            if action["type"] != "wait"
            and (len(action.get("keys", [])) != 1 or action["keys"][0] not in KEYS)
        ]
        if unsupported:
            raise PlayerResponseError(
                "Released reference plan requires unsupported native controls",
                {**self.metadata(), "unsupported_actions": unsupported},
            )
        if any(action["duration_ms"] < 0 for action in actions):
            raise ValueError("Negative reference duration")
        return actions

    @staticmethod
    def _new_state(index=0):
        return {
            "phase": index,
            "mode": "phase",
            "elapsed_ms": 0.0,
            "extension_ms": 0,
            "checks": 0,
            "directions": [],
            "pending_release": False,
        }

    def _post(self, args, payload, timeout_s):
        messages = []
        for message in payload["messages"]:
            content = []
            for part in message["content"]:
                if part["type"] == "text":
                    content.append(part)
                elif part["type"] == "image_url":
                    prefix, encoded = part["image_url"]["url"].split(",", 1)
                    if prefix != "data:image/jpeg;base64":
                        raise ValueError("Unexpected released observation encoding")
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": encoded,
                            },
                        }
                    )
                else:
                    raise ValueError("Unexpected released message content")
            messages.append({"role": message["role"], "content": content})
        response = self.client.messages.create(
            model=payload["model"],
            max_tokens=payload["max_completion_tokens"],
            messages=messages,
            timeout=timeout_s,
        )
        raw = "".join(block.text for block in response.content if block.type == "text")
        usage = (
            response.usage.model_dump()
            if getattr(response, "usage", None) is not None
            else None
        )
        self.transport = {
            "prompt_sha256": _sha(
                payload["messages"][0]["content"][0]["text"].encode()
            ),
            "response_id": getattr(response, "id", None),
            "served_model": getattr(response, "model", None),
            "usage": usage,
            "max_tokens": payload["max_completion_tokens"],
            "raw_response": raw,
            "stop_reason": getattr(response, "stop_reason", None),
        }
        if self.transport["stop_reason"] == "max_tokens":
            raise ValueError("Released controller response was truncated")
        return SimpleNamespace(
            status_code=200, json=lambda: {"choices": [{"message": {"content": raw}}]}
        )

    def _observation(self, latest):
        with Image.open(latest) as image:
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=65)
        target = SimpleNamespace(screenshot=lambda **kwargs: buffer.getvalue())
        return self.namespace["claude_viewer_screenshot"](target, _ImageSink())

    def decide(
        self, client, task, initial, latest, history, maximum=40, commanded_state=None
    ):
        if type(maximum) is not int or maximum < 1:
            raise ValueError("Positive native interaction budget required")
        signature = _sha(
            json.dumps(
                {
                    key: task.get(key)
                    for key in (
                        "task_id",
                        "action",
                        "action_sequence",
                        "action_sequence_steps",
                        "prompt",
                        "perspective",
                        "image_caption",
                    )
                },
                sort_keys=True,
            ).encode()
        )
        if not history:
            self.actions = self.validate_task(task)
            self.signature, self.state, self.issued = signature, self._new_state(), []
        if signature != self.signature or history != self.issued:
            # Rollout passes only these three fields; no opaque state is hidden
            # in human-readable reasons or reconstructed from ambiguous holds.
            normalized = [
                {key: row.get(key) for key in ("operation", "action", "reason")}
                for row in history
            ]
            if signature != self.signature or normalized != self.issued:
                raise ValueError(
                    "Reference-controller history/task changed outside this episode"
                )
        state = copy.deepcopy(self.state)
        if history:
            state["elapsed_ms"] += 812.5 if len(history) == 1 else 1000
        self.transport = None
        released = None
        operation, reason = "KEEP", "Continue the released reference phase"
        for _ in range(len(self.actions) + 5):
            if len(history) >= maximum:
                action, operation, reason = (
                    None,
                    "END",
                    "Native interaction budget exhausted; no visual-success claim",
                )
                break
            if state["phase"] >= len(self.actions):
                state["mode"] = "tail"
                action, reason = (
                    "WAIT",
                    "Reference plan exhausted; neutral observation tail, not visual-success confirmation",
                )
                break
            phase = self.actions[state["phase"]]
            if state["mode"] == "correction":
                if state["elapsed_ms"] >= 1000:
                    state["directions"].pop(0)
                    state["elapsed_ms"] = 0.0
                if state["directions"]:
                    action, operation, reason = (
                        KEYS[state["directions"][0]],
                        "CORRECT",
                        "Released one-second arrow correction after reference hold",
                    )
                    break
                state = self._new_state(state["phase"] + 1)
                operation = "STOP"
                continue
            planned = (
                phase["duration_ms"]
                if phase["type"] == "wait"
                else max(1000, phase["duration_ms"])
            )
            released_now = state["pending_release"] and state["elapsed_ms"] >= 1000
            if state["elapsed_ms"] >= planned + state["extension_ms"] or released_now:
                if state["directions"]:
                    state["mode"], state["elapsed_ms"] = "correction", 0.0
                else:
                    state = self._new_state(state["phase"] + 1)
                operation = "STOP"
                continue
            spacing = max(1500, planned / 3)
            check = (
                phase["type"] != "wait"
                and phase["keys"][0] not in self.namespace["ARROW_KEYS"]
                and history
                and state["checks"] < 3
                and state["elapsed_ms"] >= state["checks"] * spacing
            )
            if check:
                args = argparse.Namespace(
                    claude_model=MODEL,
                    claude_min_release_ms=1000,
                    claude_max_extension_ms=5000,
                    claude_timeout_s=30,
                )
                self.client = client
                try:
                    released = self.namespace["request_claude_hold_decision"](
                        args,
                        task,
                        phase,
                        state["phase"],
                        round(state["elapsed_ms"]),
                        planned,
                        state["extension_ms"],
                        self.actions[state["phase"] + 1 :],
                        self._observation(latest),
                    )
                finally:
                    self.client = None
                if "error" in released:
                    raise PlayerResponseError(
                        "Released reference controller request failed",
                        {
                            **self.metadata(),
                            "released_decision": released,
                            "transport": self.transport,
                        },
                    )
                state["checks"] += 1
                direction = released.get("adaptive_direction")
                if (
                    direction
                    and direction not in state["directions"]
                    and len(state["directions"]) < 2
                ):
                    state["directions"].append(direction)
                if released["release_now"]:
                    state["pending_release"] = True
                    if state["elapsed_ms"] >= 1000:
                        continue
                else:
                    extension = min(
                        released["extend_ms"], 10000 - state["extension_ms"]
                    )
                    state["extension_ms"] += extension
                    if extension:
                        operation = "EXTEND"
                reason = released.get("reason") or reason
            action = "WAIT" if phase["type"] == "wait" else KEYS[phase["keys"][0]]
            break
        else:
            raise ValueError("Reference phase scheduler failed to advance")
        decision = {"operation": operation, "action": action, "reason": reason}
        self.state = state
        self.issued.append(decision.copy())
        return decision, {
            "released_controller": self.metadata(),
            "released_controller_state": copy.deepcopy(state),
            "released_hold_decision": released,
            "player_transport": self.transport,
            "reference_plan_parsed": self.actions,
        }


_CONTROLLERS = weakref.WeakKeyDictionary()


def decide(client, task, initial, latest, history, maximum=40, commanded_state=None):
    """Convenience API; prefer an explicit per-episode controller instance."""
    if client not in _CONTROLLERS or not history:
        _CONTROLLERS[client] = ReleasedReferenceController()
    return _CONTROLLERS[client].decide(
        client, task, initial, latest, history, maximum, commanded_state
    )
