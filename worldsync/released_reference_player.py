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
import os
import re
import time
import weakref
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from worldsync.reference_player import PlayerResponseError

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external/PlayWorld/Agent_player/hyworld2/player.py"
SOURCE_SHA256 = "d7a90541a19c27cf4c7ed298510fb1585974c85bdd90ea39547150093b287291"
SHARED_SOURCE = ROOT / "external/PlayWorld/Agent_player/player.py"
SHARED_SOURCE_SHA256 = (
    "906e1b993b4261cbb7d56013c4fd3f52676120cc234c5bce497016dd5a7ce6d0"
)
MODEL = "claude-haiku-4-5-20251001"
PROFILE = "released-hy-reference-guided-transfer-v2"
HOLD_SERVICE_PROFILE = "released-hy-hold-check-service-strict-v1"
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
    "Raw response schema and served snapshot are now checked before the permissive released parser; this is intentional fail-closed hardening.",
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


def canonical_actions(task, source_path=SHARED_SOURCE):
    """Shared released action DSL, exact milliseconds, no HY counts/padding.

    Unsupported raw action descriptions are never interpreted as camera inputs.
    The root timeline owns phase boundaries and angle/speed calibration.
    """
    raw = Path(source_path).read_bytes()
    if _sha(raw) != SHARED_SOURCE_SHA256:
        raise ValueError("Shared released action DSL source changed")
    tree = ast.parse(raw.decode())
    namespace = {"re": re, "dataclass": dataclass, "__name__": __name__}
    nodes = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id in {"ACTION_RE", "WAIT_RE", "KEY_ALIASES"}
                for target in node.targets
            )
            or isinstance(node, ast.ClassDef)
            and node.name == "ActionStep"
            or isinstance(node, ast.FunctionDef)
            and node.name == "parse_action_step"
        ):
            nodes.append(node)
    if len(nodes) != 5:
        raise ValueError("Shared action DSL definitions are incomplete")
    compiled = compile(
        ast.Module(body=nodes, type_ignores=[]),
        str(source_path),
        "exec",
        dont_inherit=True,
    )
    exec(compiled, namespace)  # noqa: S102 -- SHA-pinned shared parser/class/constants only
    steps = task.get("action_sequence_steps") or task.get("action_sequence") or []
    if isinstance(steps, str):
        steps = [step.strip() for step in steps.split(";") if step.strip()]
    if not isinstance(steps, (list, tuple)) or (not steps and task.get("action")):
        raise PlayerResponseError(
            "No canonical shared DSL plan; raw action fallback is unsupported",
            {"shared_dsl_source_sha256": SHARED_SOURCE_SHA256},
        )
    result = []
    for index, step in enumerate(steps):
        parsed = namespace["parse_action_step"](step)
        # These pitch aliases are explicit in the pinned HY native key map;
        # the shared generic parser intentionally leaves them uninterpreted.
        pitch_aliases = {"U": "ArrowUp", "J": "ArrowDown"}
        keys = [
            "Key" + key.upper()
            if key.upper() in {"W", "A", "S", "D"}
            else pitch_aliases.get(key.upper(), key)
            for key in parsed.keys
        ]
        if parsed.kind != "wait" and (len(keys) != 1 or keys[0] not in KEYS):
            raise PlayerResponseError(
                "Canonical plan requires unsupported native controls",
                {
                    "raw_step": step,
                    "keys": keys,
                    "shared_dsl_source_sha256": SHARED_SOURCE_SHA256,
                },
            )
        result.append(
            {
                "type": parsed.kind,
                "keys": keys,
                "duration_ms": parsed.duration_ms,
                "raw": parsed.raw,
                "index": index,
                "rotation_extra_ms": 0,
                "source": "released shared action DSL; no HY timing conversion",
                "native_pitch_aliases": [
                    {"source": key, "target": pitch_aliases[key.upper()]}
                    for key in parsed.keys
                    if key.upper() in pitch_aliases
                ],
            }
        )
    return result


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
        raw_token_limit = os.environ.get("WORLDSYNC_HOLD_MAX_TOKENS", "0") or "0"
        try:
            self.token_limit_override = int(raw_token_limit)
        except ValueError:
            raise ValueError("WORLDSYNC_HOLD_MAX_TOKENS must be a nonnegative integer") from None
        if self.token_limit_override < 0:
            raise ValueError("WORLDSYNC_HOLD_MAX_TOKENS must be a nonnegative integer")
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
            "output_token_policy": {
                "released_max_tokens": 60,
                "override": self.token_limit_override,
                "effective_max_tokens": max(60, self.token_limit_override),
                "deviation_from_released_cap": self.token_limit_override > 60,
            },
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

    def hold_service_metadata(self):
        result = self.metadata()
        legacy_only = (
            "Durations use generated time",
            "First live check observes",
            "Released HY action-field priority",
            "Neutral observation tail ends",
            "A40interaction paper limit",
        )
        result.update(
            profile=HOLD_SERVICE_PROFILE,
            scheduler_policy=None,
            low_level_service_only=True,
            hy_action_parser_and_scheduler_used=False,
            shared_dsl_source_sha256=SHARED_SOURCE_SHA256,
        )
        result["remaining_differences"] = [
            text for text in DIFFERENCES if not text.startswith(legacy_only)
        ]
        result["caller_responsibilities"] = [
            "Canonical phase parsing, clock and angle calibration",
            "Check cadence, minimum hold/release, phase progression and rotation-release policy",
            "Global interaction budget and extension/correction application",
        ]
        return result

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
        # The released payload caps output at 60 tokens, which Haiku's fenced JSON plus reason exceeds (stop_reason
        # max_tokens, unparseable). WORLDSYNC_HOLD_MAX_TOKENS raises the cap explicitly; default keeps the released value.
        max_tokens = max(payload["max_completion_tokens"], self.token_limit_override)
        response = self.client.messages.create(
            model=payload["model"],
            max_tokens=max_tokens,
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
            "max_tokens": max_tokens,
            "released_max_tokens": payload["max_completion_tokens"],
            "raw_response": raw,
            "stop_reason": getattr(response, "stop_reason", None),
        }
        if self.transport["served_model"] != MODEL:
            raise ValueError(
                "Controller served model does not match the requested Haiku snapshot"
            )
        if self.transport["stop_reason"] == "max_tokens":
            raise ValueError("Released controller response was truncated")
        parsed = self.namespace["extract_json_object"](raw)
        expected = {"release_now", "extend_ms", "adaptive_direction", "reason"}
        if (
            set(parsed) != expected
            or type(parsed["release_now"]) is not bool
            or type(parsed["extend_ms"]) is not int
            or parsed["extend_ms"] < 0
            or not isinstance(parsed["reason"], str)
            or not isinstance(parsed["adaptive_direction"], str)
        ):
            raise ValueError("Controller JSON does not match the released field schema")
        direction = parsed["adaptive_direction"].strip().lower()
        if (
            direction != "none"
            and direction not in self.namespace["CLAUDE_ADAPTIVE_KEY_MAP"]
        ):
            raise ValueError("Controller returned an unsupported adaptive direction")
        return SimpleNamespace(
            status_code=200, json=lambda: {"choices": [{"message": {"content": raw}}]}
        )

    def _observation(self, latest):
        with Image.open(latest) as image:
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=65)
        target = SimpleNamespace(screenshot=lambda **kwargs: buffer.getvalue())
        return self.namespace["claude_viewer_screenshot"](target, _ImageSink())

    def check_hold(
        self, client, task, actions, phase_index, elapsed_ms, extension_ms, latest
    ):
        """Released visual hold request on an explicit caller-owned phase.

        Does not parse HY action counts, pad durations, advance phases, or apply
        release/extension/correction policies. The caller owns those decisions.
        """
        if not isinstance(actions, list) or not actions:
            raise ValueError("Explicit nonempty canonical actions required")
        if type(phase_index) is not int or not 0 <= phase_index < len(actions):
            raise ValueError("Invalid canonical phase index")
        for action in actions:
            keys = action.get("keys", []) if isinstance(action, dict) else None
            if (
                not isinstance(action, dict)
                or action.get("type") not in {"hold", "wait"}
                or type(action.get("duration_ms")) is not int
                or action["duration_ms"] < 0
                or not isinstance(keys, (list, tuple))
                or (action["type"] == "wait" and keys)
                or (
                    action["type"] == "hold" and (len(keys) != 1 or keys[0] not in KEYS)
                )
            ):
                raise ValueError("Invalid or unsupported explicit canonical action")
        if (
            type(elapsed_ms) is not int
            or elapsed_ms < 0
            or type(extension_ms) is not int
            or not 0 <= extension_ms <= 10000
        ):
            raise ValueError("Invalid explicit phase clock/extension")
        if self.client is not None:
            raise RuntimeError(
                "Hold checks must be serial within a controller instance"
            )
        args = argparse.Namespace(
            claude_model=MODEL,
            claude_min_release_ms=1000,
            claude_max_extension_ms=5000,
            claude_timeout_s=30,
        )
        self.transport = None
        self.client = client
        try:
            result = self.namespace["request_claude_hold_decision"](
                args,
                task,
                actions[phase_index],
                phase_index,
                elapsed_ms,
                actions[phase_index]["duration_ms"],
                extension_ms,
                actions[phase_index + 1 :],
                self._observation(latest),
            )
        finally:
            self.client = None
        metadata = {
            "released_controller": self.hold_service_metadata(),
            "player_transport": self.transport,
            "released_hold_decision": result,
            "explicit_phase_index": phase_index,
            "explicit_elapsed_ms": elapsed_ms,
            "explicit_planned_ms": actions[phase_index]["duration_ms"],
            "shared_dsl_source_sha256": SHARED_SOURCE_SHA256,
        }
        if "error" in result:
            raise PlayerResponseError("Released hold service request failed", metadata)
        return result, metadata

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
