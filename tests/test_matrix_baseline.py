from types import SimpleNamespace

from PIL import Image
import pytest

from worldsync.matrix_baseline import decide, native_action, player_prompt


def test_matrix_native_action_axes_and_invalid_controls():
    assert native_action("LEFT") == {"keyboard": [0, 0, 0, 0, 0, 0], "mouse": [0, -0.1]}
    assert native_action("UP")["mouse"] == [0.1, 0]
    assert native_action("W")["keyboard"] == [1, 0, 0, 0, 0, 0]
    assert native_action("WAIT") == {"keyboard": [0, 0, 0, 0, 0, 0], "mouse": [0, 0]}
    with pytest.raises(ValueError):
        native_action("W+RIGHT")


def test_reconstructed_player_retains_reference_without_leaking_judge_questions(tmp_path):
    task = {"prompt": "Turn and return", "action_sequence_steps": ["hold(LEFT,1800ms)"],
            "questions": [{"question": "private scoring rubric"}]}
    prompt = player_prompt(task, [], 40)
    assert "hold(LEFT,1800ms)" in prompt
    assert "private scoring rubric" not in prompt
    image = tmp_path / "frame.png"
    Image.new("RGB", (16, 16)).save(image)
    captured = {}
    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text='{"operation":"END","action":null,"reason":"returned"}')],
            id="test", model="haiku-test", usage=SimpleNamespace(model_dump=lambda: {}))
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    decision, metadata = decide(client, "haiku-test", task, image, image, [], 40)
    assert decision["operation"] == "END"
    assert metadata["response_id"] == "test"
    assert len([block for block in captured["messages"][0]["content"] if block["type"] == "image"]) == 2
