from types import SimpleNamespace

import pytest
pytest.importorskip("pydantic")
from PIL import Image
from pydantic import ValidationError

from worldsync.player import Decision, KEY_MAP, OpenAIPlayer


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
