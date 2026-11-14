from types import SimpleNamespace

import pytest
pytest.importorskip("pydantic")
from PIL import Image
from pydantic import ValidationError

from worldsync.player import AnthropicPlayer, Decision, KEY_MAP, OpenAIPlayer


def test_player_uses_observations_and_validated_actions(tmp_path):
    image = tmp_path / "view.png"
    Image.new("RGB", (16, 16)).save(image)
    captured = {}

    def parse(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_parsed=Decision(action="RIGHT", blocks=2, reason="inspect the scene"), id="test", model="gpt-6.1-sol", usage=None)

    client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    decision, metadata = OpenAIPlayer(client=client).decide(image, image, "Turn right, then return", [], 40)
    assert KEY_MAP[decision.action] == "L"
    assert captured["store"] is False
    assert captured["reasoning"] == {"effort": "low"}
    assert len([item for item in captured["input"][0]["content"] if item["type"] == "input_image"]) == 2
    assert metadata["model"] == "gpt-6.1-sol"


def test_controller_rejects_unbounded_or_unknown_actions():
    with pytest.raises(ValidationError):
        Decision(action="SHELL", blocks=2, reason="invalid")
    with pytest.raises(ValidationError):
        Decision(action="WAIT", blocks=1000, reason="too long")


def test_claude_uses_validated_outputs_without_forcing_thinking_off(tmp_path):
    image = tmp_path / "view.png"
    Image.new("RGB", (16, 16)).save(image)
    captured = {}

    def parse(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(parsed_output=Decision(action="WAIT", blocks=1, reason="observe"), id="test", model="claude-opus-5-5", usage=None)

    client = SimpleNamespace(messages=SimpleNamespace(parse=parse))
    decision, _ = AnthropicPlayer(client=client).decide(image, image, "Observe the scene", [], 40)
    assert decision.action == "WAIT"
    assert captured["thinking"] == {"type": "adaptive"}
    assert captured["output_format"] is Decision
    assert "tool_choice" not in captured
