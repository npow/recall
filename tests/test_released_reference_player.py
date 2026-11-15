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
    canonical_actions,
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


def test_output_cap_override_is_explicit_and_frozen_per_controller(monkeypatch, observation):
    monkeypatch.setenv('WORLDSYNC_HOLD_MAX_TOKENS','400')
    controller=ReleasedReferenceController()
    monkeypatch.setenv('WORLDSYNC_HOLD_MAX_TOKENS','800')
    client=FakeClient()
    data=task(action_sequence_steps=['hold(W,3000ms)'])
    _,metadata=controller.check_hold(client,data,canonical_actions(data),0,0,0,observation)
    assert client.calls[0]['max_tokens']==400
    assert metadata['player_transport']['released_max_tokens']==60
    assert metadata['player_transport']['max_tokens']==400
    policy=controller.hold_service_metadata()['output_token_policy']
    assert policy['effective_max_tokens']==400 and policy['deviation_from_released_cap']


@pytest.mark.parametrize('value',['-1','not-an-integer','60.5'])
def test_invalid_output_cap_rejected_before_any_api_or_gpu(monkeypatch,value):
    monkeypatch.setenv('WORLDSYNC_HOLD_MAX_TOKENS',value)
    with pytest.raises(ValueError,match='nonnegative integer'):
        ReleasedReferenceController()


def test_raised_cap_does_not_accept_truncated_response(monkeypatch,observation):
    monkeypatch.setenv('WORLDSYNC_HOLD_MAX_TOKENS','400')
    class TruncatedClient(FakeClient):
        def create(self,**kwargs):
            result=super().create(**kwargs)
            result.stop_reason='max_tokens'
            return result
    data=task(action_sequence_steps=['hold(W,3000ms)'])
    with pytest.raises(PlayerResponseError):
        ReleasedReferenceController().check_hold(TruncatedClient(),data,canonical_actions(data),0,0,0,observation)


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


def test_canonical_player_uses_released_hold_service_end_to_end(observation):
    from worldsync.lingbot_released_baseline import BaselinePlayer, load_task
    controller, client = ReleasedReferenceController(), FakeClient()
    player = BaselinePlayer(controller)
    data = load_task('if', 'IF001')
    history = []
    first, first_meta = player.decide(client, data, observation, observation, history)
    history.append(first)
    assert first_meta['bootstrap'] and not client.calls
    second, meta = player.decide(client, data, observation, observation, history)
    assert len(client.calls) == player.requests == meta['agent_interaction_increment'] == 1
    assert meta['released_controller']['hy_action_parser_and_scheduler_used'] is False
    assert meta['explicit_phase_index'] == 0 and meta['explicit_elapsed_ms'] == 0
    assert meta['released_hold_decision']['adaptive_direction'] is None
    assert meta['player_transport']['served_model'] == MODEL
    assert len(meta['native_camera_steps']) == 4


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


def test_canonical_shared_dsl_exact_duration_without_hy_fallback(tmp_path):
    data = task(
        action="right*4 -> left*4",
        action_sequence_steps=[
            "hold(Right,1800ms)",
            "hold(Left,1800ms)",
            "wait(500ms)",
        ],
    )
    actions = canonical_actions(data)
    assert [a["duration_ms"] for a in actions] == [1800, 1800, 500]
    assert [a["keys"] for a in actions] == [["ArrowRight"], ["ArrowLeft"], []]
    assert all(a["rotation_extra_ms"] == 0 for a in actions)
    assert canonical_actions(task(action_sequence_steps=["hold(w,450ms)"]))[0][
        "keys"
    ] == ["KeyW"]
    pitch = canonical_actions(
        task(action_sequence_steps=["hold(U,450ms)", "hold(J,900ms)"])
    )
    assert [step["keys"] for step in pitch] == [["ArrowUp"], ["ArrowDown"]]
    assert pitch[0]["native_pitch_aliases"] == [{"source": "U", "target": "ArrowUp"}]
    with pytest.raises(PlayerResponseError, match="raw action fallback"):
        canonical_actions(task(action_sequence_steps=[]))
    with pytest.raises(PlayerResponseError, match="unsupported native controls"):
        canonical_actions(task(action_sequence_steps=["hold(SPACE,500ms)"]))
    changed = tmp_path / "shared.py"
    changed.write_text("# changed")
    with pytest.raises(ValueError, match="DSL source changed"):
        canonical_actions(data, changed)


def test_low_level_service_uses_explicit_canonical_phase_and_no_scheduler(observation):
    controller, client = ReleasedReferenceController(), FakeClient()
    data = task(action_sequence_steps=["hold(W,4500ms)", "hold(Right,1800ms)"])
    actions = canonical_actions(data)
    released, metadata = controller.check_hold(
        client, data, actions, 0, 1250, 500, observation
    )
    prompt = client.calls[0]["messages"][0]["content"][0]["text"]
    assert (
        "Elapsed hold: 1250ms; original planned hold: 4500ms; extension already granted: 500ms"
        in prompt
    )
    assert "Current held key(s): ['KeyW']" in prompt
    assert (
        released["release_now"] is False
        and controller.state is None
        and not controller.issued
    )
    assert metadata["released_controller"]["scheduler_policy"] is None
    assert (
        metadata["released_controller"]["hy_action_parser_and_scheduler_used"] is False
    )
    assert controller.client is None


@pytest.mark.parametrize(
    "change",
    [
        {"release_now": "false"},
        {"extend_ms": "100"},
        {"extend_ms": True},
        {"adaptive_direction": None},
        {"adaptive_direction": "w"},
        {"reason": 10},
        {"extra": 1},
    ],
)
def test_strict_schema_prevents_released_parser_coercions(observation, change):
    controller = ReleasedReferenceController()
    response = {
        "release_now": False,
        "extend_ms": 0,
        "adaptive_direction": "none",
        "reason": "continue",
        **change,
    }
    client = FakeClient([response])
    with pytest.raises(PlayerResponseError, match="service request failed") as error:
        controller.check_hold(
            client, task(), canonical_actions(task()), 0, 1250, 0, observation
        )
    assert (
        controller.client is None
        and error.value.metadata["player_transport"]["raw_response"]
    )


def test_service_rejects_served_model_mismatch_and_invalid_inputs(observation):
    controller, client = ReleasedReferenceController(), FakeClient()
    original = client.messages.create

    def wrong_model(**kwargs):
        response = original(**kwargs)
        response.model = "not-haiku"
        return response

    client.messages.create = wrong_model
    with pytest.raises(PlayerResponseError, match="service request failed") as error:
        controller.check_hold(
            client, task(), canonical_actions(task()), 0, 1250, 0, observation
        )
    assert error.value.metadata["player_transport"]["served_model"] == "not-haiku"
    assert controller.client is None
    count = len(client.calls)
    with pytest.raises(ValueError, match="phase index"):
        controller.check_hold(
            client, task(), canonical_actions(task()), 3, 1250, 0, observation
        )
    with pytest.raises(ValueError, match="phase clock"):
        controller.check_hold(
            client, task(), canonical_actions(task()), 0, 1.5, 0, observation
        )
    assert len(client.calls) == count
