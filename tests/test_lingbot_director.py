import json
from types import SimpleNamespace

import pytest

from worldsync.lingbot_director import ALLOWED_TASK_FIELDS, Director, director_inputs, view_note

TASK = {"task_id": "OE999", "image_caption": "A hand paints a blue area on paper.", "prompt": "Turn left 90 degrees, then turn back.",
        "action_sequence_steps": ["hold(Left,1800ms)", "hold(Right,1800ms)"],
        "questions": [{"question": "SECRET RUBRIC QUESTION"}], "oe_dynamicity_policy": "SECRET POLICY",
        "non_task_specific_evaluation_policy": "SECRET POLICY 2"}


class FakeClient:
    def __init__(self, fail=False):
        self.requests, self.fail = [], fail
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.fail:
            raise RuntimeError("api down")
        n = len(self.requests)
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", input={"world_state": f"state {n}", "prompt": f"text {n}"})])


def test_firewall_whitelist_only():
    assert set(director_inputs(TASK)) == set(ALLOWED_TASK_FIELDS)
    client = FakeClient()
    director = Director(client, TASK, "unseen", interval=2)
    director.initial()
    director.for_chunk(1, "LEFT", {"heading_degrees": -60, "position_normalized_units": [0, 0, 0]}, 0.8)
    director.for_chunk(2, "LEFT", {"heading_degrees": -90, "position_normalized_units": [0, 0, 0]}, 1.8)
    sent = json.dumps([r["messages"] for r in client.requests])
    assert "SECRET" not in sent and "questions" not in sent and "policy" not in sent.lower()
    assert "SECRET" not in json.dumps([r["system"] for r in client.requests])


def test_static_calls_llm_once():
    client = FakeClient()
    director = Director(client, TASK, "static")
    first = director.initial()
    assert director.for_chunk(0, "WAIT", None, 0.0) == (first, False)
    assert director.for_chunk(7, "LEFT", None, 6.8) == (first, False)
    assert len(client.requests) == 1


def test_perchunk_refreshes_on_interval_and_action_change():
    client = FakeClient()
    director = Director(client, TASK, "perchunk", interval=4)
    director.initial()
    assert director.for_chunk(0, "WAIT", None, 0.0)[1] is False
    assert director.for_chunk(1, "WAIT", None, 0.8)[1] is True       # first refresh after the initial text
    assert director.for_chunk(2, "WAIT", None, 1.8)[1] is False
    assert director.for_chunk(3, "LEFT", None, 2.8)[1] is True       # action changed
    assert director.for_chunk(4, "LEFT", None, 3.8)[1] is False
    assert director.for_chunk(7, "LEFT", None, 6.8)[1] is True       # interval elapsed
    assert "camera_state" not in client.requests[-1]["messages"][0]["content"]


def test_unseen_reports_view():
    client = FakeClient()
    director = Director(client, TASK, "unseen", interval=1)
    director.initial()
    director.for_chunk(1, "LEFT", {"heading_degrees": -90, "position_normalized_units": [0, 0, 0]}, 0.8)
    assert "out of frame" in client.requests[-1]["messages"][0]["content"]
    assert "in frame" in view_note({"heading_degrees": 3, "position_normalized_units": [0, 0, 0]})


def test_llm_failure_falls_back_to_caption():
    director = Director(FakeClient(fail=True), TASK, "perchunk")
    assert director.initial() == TASK["image_caption"]
    assert director.for_chunk(3, "LEFT", None, 2.8) == (TASK["image_caption"], False)
    assert director.metadata()["errors"] == 2


def test_rejects_off_mode():
    with pytest.raises(ValueError):
        Director(FakeClient(), TASK, "off")


def test_preset_director_schedule(tmp_path):
    from worldsync.lingbot_director import PresetDirector
    path = tmp_path / "texts.json"
    path.write_text(json.dumps({"cases": {"outsight:OE1": {"static": "s", "scheduled": [[0, "a"], [2, "b"]]},
                                          "if:IF1": {"static": "only"}}}))
    director = PresetDirector(path, "outsight", "OE1", "unseen")
    assert director.initial() == "a"
    assert director.for_chunk(1, "LEFT", None, 0.8) == ("a", False)
    assert director.for_chunk(2, "RIGHT", None, 1.8) == ("b", True)
    assert director.for_chunk(3, "RIGHT", None, 2.8) == ("b", False)
    assert PresetDirector(path, "outsight", "OE1", "static").initial() == "s"
    assert PresetDirector(path, "if", "IF1", "perchunk").for_chunk(5, "W", None, 4.8) == ("only", False)
    with pytest.raises(ValueError):
        PresetDirector(path, "gc", "GC1", "static")


def test_committed_dev_texts_cover_dev_cases():
    from pathlib import Path
    from worldsync.lingbot_director import PresetDirector
    path = Path(__file__).resolve().parents[1] / "reproduction/director-dev-texts.json"
    cases = json.loads(path.read_text())["cases"]
    assert len(cases) == 20
    for key in cases:
        split, task = key.split(":")
        for mode in ("static", "unseen"):
            assert PresetDirector(path, split, task, mode).initial()
