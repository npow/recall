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


def test_held_seconds_counts_consecutive_chunks_with_short_first_chunk():
    from worldsync.reference_player import held_seconds
    assert held_seconds([]) == 0.0
    w = {"action": "W"}
    assert held_seconds([w] * 3) == pytest.approx(2.8125)
    assert held_seconds([w, {"action": "S"}, {"action": "S"}]) == 2.0
    assert held_seconds([w, {"action": None}]) == 0.0


def test_plan_steps_and_elapsed_time():
    from worldsync.reference_player import plan_steps, elapsed_seconds
    steps = plan_steps({"action_sequence_steps": ["hold(Right,1800ms)", "wait(450ms)", "hold(W,2250ms)", "bogus"]})
    assert steps == [{"action": "RIGHT", "seconds": 1.8}, {"action": "WAIT", "seconds": 0.45}, {"action": "W", "seconds": 2.25}]
    assert elapsed_seconds([]) == 0.0 and elapsed_seconds([{}] * 3) == pytest.approx(2.8125)


def test_early_end_is_challenged_once(tmp_path):
    image = tmp_path / "i.jpg"
    Image.new("RGB", (16, 16)).save(image)
    task = {"prompt": "Walk", "action_sequence_steps": ["hold(W,7200ms)"]}
    prompts = []
    answers = iter([{"operation": "END", "action": "NONE", "reason": "looks done"},
                    {"operation": "KEEP", "action": "W", "reason": "plan not finished"}])
    def create(**kwargs):
        prompts.append(kwargs["messages"][0]["content"][0]["text"])
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="choose_action", input=next(answers))],
            id=f"r{len(prompts)}", model="m", usage=SimpleNamespace(model_dump=lambda: {}))
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    history = [{"operation": "KEEP", "action": "W", "reason": "x"}] * 3
    decision, metadata = decide(client, task, image, image, history)
    assert decision["operation"] == "KEEP" and len(prompts) == 2
    assert "controller_note" in prompts[1] and "controller_note" not in prompts[0]
    assert metadata["first_attempt_rejected_end"] == "looks done"
