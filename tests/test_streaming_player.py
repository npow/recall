from types import SimpleNamespace as NS
import threading

import pytest

from worldsync.streaming_player import StreamingDecision, action_prefix


def test_prefix_requires_complete_action_and_never_speculates_end():
    assert action_prefix('{"operation":"KEEP","action":"RIGH') is None
    assert action_prefix('{"operation":"END","action":"NONE"') is None
    assert action_prefix('{"operation":"KEEP","action":"RIGHT",') == {"operation": "KEEP", "action": "RIGHT"}
    assert action_prefix('{"reason":"escaped \\"action\\":\\"RIGHT\\"') is None


def client_and_release(final_action="RIGHT", fail=False):
    release = threading.Event()
    response = NS(content=[NS(type="tool_use", name="choose_action", input={
        "operation": "KEEP", "action": final_action, "reason": "Complete original explanation"})],
        id="response1", model="unchanged-model", usage=NS(model_dump=lambda: {"output_tokens": 30}))
    captured = []
    class Stream:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def __iter__(self):
            yield NS(type="content_block_start", index=0, content_block=NS(type="tool_use", name="choose_action"))
            yield NS(type="content_block_delta", index=0, delta=NS(type="input_json_delta",
                partial_json='{"operation":"KEEP","action":"RIGHT",'))
            assert release.wait(5)
            if fail:
                raise ConnectionError("Truncated stream")
        def get_final_message(self): return response
        def close(self): release.set()
    def stream(**kwargs):
        captured.append(kwargs)
        return Stream()
    return NS(messages=NS(stream=stream)), release, captured


def test_original_controller_request_and_full_metadata_survive_overlap(tmp_path):
    from PIL import Image
    image = tmp_path / "scene.jpg"
    Image.new("RGB", (8, 8)).save(image)
    client, release, captured = client_and_release()
    pending = StreamingDecision(client, {"prompt": "turn right"}, image, image, [])
    try:
        action, metadata = pending.action()
        assert action["action"] == "RIGHT" and metadata["player_response_pending"]
        assert not pending.future.done()  # Generation can start while explanation is pending.
        release.set()
        final, metadata = pending.finish()
        assert final["reason"] == "Complete original explanation"
        assert metadata["response_id"] == "response1"
        assert metadata["usage"] == {"output_tokens": 30}
        assert metadata["player_stream_timing"]["generation_started_before_response_completed"]
        assert captured[0]["model"] == "claude-haiku-4-5-20251001"
        assert "temperature" not in captured[0]
    finally:
        pending.close()


@pytest.mark.parametrize("fail,final_action,error", [(True, "RIGHT", ConnectionError), (False, "LEFT", RuntimeError)])
def test_stream_failure_or_changed_final_action_invalidates_speculation(tmp_path, fail, final_action, error):
    from PIL import Image
    image = tmp_path / "scene.jpg"
    Image.new("RGB", (8, 8)).save(image)
    client, release, _ = client_and_release(final_action, fail)
    pending = StreamingDecision(client, {"prompt": "turn right"}, image, image, [])
    try:
        assert pending.action()[0]["action"] == "RIGHT"
        release.set()
        with pytest.raises(error):
            pending.finish()
    finally:
        pending.close()


def test_premature_end_challenge_and_requests_match_original_controller(tmp_path):
    from PIL import Image
    from worldsync.reference_player import decide
    image = tmp_path / "scene.jpg"
    Image.new("RGB", (8, 8)).save(image)
    task = {"prompt": "turn right", "action_sequence_steps": ["hold(Right,5000ms)"]}
    def response(index):
        decision = ({"operation": "END", "action": "NONE", "reason": "looks done"} if index == 0 else
                    {"operation": "EXTEND", "action": "RIGHT", "reason": "continue turning"})
        return NS(content=[NS(type="tool_use", name="choose_action", input=decision)],
                  id=f"response{index}", model="model", usage=NS(model_dump=lambda: {"output_tokens": 25}))
    standard_requests, streaming_requests = [], []
    def create(**kwargs):
        standard_requests.append(kwargs)
        return response(len(standard_requests) - 1)
    expected, expected_metadata = decide(NS(messages=NS(create=create)), task, image, image, [])
    class Stream:
        def __init__(self, message): self.message = message
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def __iter__(self): return iter(())
        def get_final_message(self): return self.message
        def close(self): pass
    def stream(**kwargs):
        streaming_requests.append(kwargs)
        return Stream(response(len(streaming_requests) - 1))
    pending = StreamingDecision(NS(messages=NS(stream=stream)), task, image, image, [])
    try:
        actual, metadata = pending.action()
        assert actual == expected
        assert streaming_requests == standard_requests  # Same model, schema, images and both prompts.
        assert {key: metadata[key] for key in expected_metadata} == expected_metadata
        assert metadata["first_attempt_rejected_end"] == "looks done"
    finally:
        pending.close()
