from types import SimpleNamespace

from PIL import Image
import pytest

from worldsync.reference_player import decide


def test_reference_controller_uses_native_timing_and_excludes_judge_questions(tmp_path):
    image = tmp_path / "initial.jpg"
    Image.new("RGB", (16, 16)).save(image)
    task = {"prompt": "Walk back", "action_sequence_steps": ["hold(S,1800ms)"],
            "questions": ["Hidden scoring question"]}
    captured = {}
    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="choose_action", input={"operation":"KEEP","action":"S","reason":"returning"})],
            id="test", model="haiku-test", usage=SimpleNamespace(model_dump=lambda: {}))
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    decision, _ = decide(client, task, image, image, [], commanded_state={"heading_degrees": 120})
    assert decision["action"] == "S"
    text = captured["messages"][0]["content"][0]["text"]
    assert "13 frames at 16 FPS" in text and "hold(S,1800ms)" in text
    assert "Hidden scoring question" not in text
    assert "priority over the reference plan" in text
    assert "Walking backward (S) does not turn around" in text
    assert '"heading_degrees": 120' in text
    assert captured["model"] == "claude-haiku-4-5-20251001"
    assert captured["tools"][0]["strict"] is True
    assert captured["tools"][0]["input_schema"]["properties"]["action"]["enum"] == ["A", "D", "DOWN", "LEFT", "RIGHT", "S", "UP", "W", "WAIT", "NONE"]
