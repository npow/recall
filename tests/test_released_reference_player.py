import copy
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from worldsync.reference_player import PlayerResponseError
from worldsync.released_reference_player import (
    MODEL,
    PROFILE,
    ReleasedReferenceController,
)


class FakeClient:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        raw = (
            self.responses.pop(0)
            if self.responses
            else {
                "release_now": False,
                "extend_ms": 0,
                "adaptive_direction": "none",
                "reason": "continue",
            }
        )
        text = raw if isinstance(raw, str) else json.dumps(raw)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=None,
            model=MODEL,
            id="fake-response",
            stop_reason="end_turn",
        )


@pytest.fixture
def observation(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (832, 464), color=(80, 100, 120)).save(path)
    return path


def task(**overrides):
    return {
        "task_id": "IF001",
        "action": "w*2 -> s*1",
        "action_sequence_steps": ["hold(W,100ms)"],
        "prompt": "Walk forward then backward",
        "image_caption": "A person on a path",
        "perspective": "third person",
        **overrides,
    }


def call(controller, client, history, image, data=None, maximum=40):
    decision, meta = controller.decide(
        client, data or task(), image, image, history, maximum=maximum
    )
    history.append(decision)
    return decision, meta


def test_exact_released_payload_parser_and_sdk_transport(observation):
    controller, client, history = ReleasedReferenceController(), FakeClient(), []
    first, meta = call(controller, client, history, observation)
    assert first["action"] == "W" and not client.calls
    assert meta["reference_plan_parsed"][0]["duration_ms"] == 3000
    second, meta = call(controller, client, history, observation)
    request = client.calls[0]
    assert (
        request["model"] == MODEL
        and request["max_tokens"] == 60
        and request["timeout"] == 30
    )
    assert "tools" not in request
    assert len(request["messages"][0]["content"]) == 2
    text = request["messages"][0]["content"][0]["text"]
    assert text.startswith(
        "You are the visual closed-loop controller for a Tencent HY WorldPlay benchmark."
    )
    assert '"release_now":false,"extend_ms":0,"adaptive_direction":"none"' in text
    assert "Elapsed hold: 812ms; original planned hold: 3000ms" in text
    assert "Perspective: third person" in text and "choose_action" not in text
    assert request["messages"][0]["content"][1]["source"]["media_type"] == "image/jpeg"
    assert meta["released_controller"]["profile"] == PROFILE and second["action"] == "W"
    assert meta["released_hold_decision"]["reason"] == "continue"


def test_phase_timing_arrows_no_api_tail_and_budget_end(observation):
    controller, client, history = ReleasedReferenceController(), FakeClient(), []
    data = task(action="right*1")
    assert (
        controller.validate_task(data)[0]["duration_ms"] == 3000
    )  # 1.5s unit plus1.5s padding.
    decisions = [
        call(controller, client, history, observation, data, maximum=6)[0]
        for _ in range(6)
    ]
    assert [row["action"] for row in decisions] == ["RIGHT"] * 4 + ["WAIT"] * 2
    assert not client.calls
    end, _ = call(controller, client, history, observation, data, maximum=6)
    assert end["operation"] == "END" and end["action"] is None
    assert "no visual-success claim" in end["reason"]


def test_release_minimum_and_postrelease_correction(observation):
    controller = ReleasedReferenceController()
    client = FakeClient(
        [
            {
                "release_now": True,
                "extend_ms": 5000,
                "adaptive_direction": "left",
                "reason": "boundary reached",
            }
        ]
    )
    history = []
    call(controller, client, history, observation)
    early, meta = call(controller, client, history, observation)
    assert (
        early["action"] == "W" and meta["released_controller_state"]["pending_release"]
    )
    assert meta["released_controller_state"]["extension_ms"] == 0
    correction, _ = call(controller, client, history, observation)
    assert correction["action"] == "LEFT" and correction["operation"] == "CORRECT"
    following, _ = call(controller, client, history, observation)
    assert following["action"] == "S" and following["operation"] == "STOP"


def test_extension_check_and_total_limits(observation):
    controller = ReleasedReferenceController()
    client = FakeClient(
        [
            {
                "release_now": False,
                "extend_ms": 50000,
                "adaptive_direction": "none",
                "reason": "extend",
            }
        ]
        * 3
    )
    history = []
    data = task(action="w*1")
    for _ in range(9):
        _, meta = call(controller, client, history, observation, data)
    assert len(client.calls) == 3
    assert meta["released_controller_state"]["checks"] == 3
    assert meta["released_controller_state"]["extension_ms"] == 10000


@pytest.mark.parametrize(
    "data",
    [
        task(action="interact(grip handle)"),
        task(action="space"),
        task(action="", action_sequence_steps=["hold(W+A,1000ms)"]),
    ],
)
def test_unsupported_avatar_or_simultaneous_controls_fail_without_api(
    data, observation
):
    client = FakeClient()
    with pytest.raises(PlayerResponseError, match="unsupported native controls"):
        ReleasedReferenceController().decide(client, data, observation, observation, [])
    assert not client.calls


def test_malformed_released_response_failclosed_and_transactional(observation):
    controller, client, history = (
        ReleasedReferenceController(),
        FakeClient(["not JSON"]),
        [],
    )
    call(controller, client, history, observation)
    before = copy.deepcopy(controller.state)
    with pytest.raises(PlayerResponseError, match="request failed") as error:
        call(controller, client, history, observation)
    assert "error" in error.value.metadata["released_decision"]
    assert (
        controller.state == before
        and controller.client is None
        and len(controller.issued) == 1
    )


def test_episode_history_guards_reset_and_pinned_source(observation, tmp_path):
    controller, client, history = ReleasedReferenceController(), FakeClient(), []
    call(controller, client, history, observation)
    altered = [{**history[0], "action": "D"}]
    with pytest.raises(ValueError, match="history/task"):
        controller.decide(client, task(), observation, observation, altered)
    assert not client.calls
    reset, _ = controller.decide(
        client, task(action="a*1"), observation, observation, []
    )
    assert reset["action"] == "A"
    source = tmp_path / "changed.py"
    source.write_text("# changed")
    with pytest.raises(ValueError, match="source differs"):
        ReleasedReferenceController(source)
