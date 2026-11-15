"""Overlap a streamed action's trailing explanation with world generation.

The existing controller builds and validates every request/response. No prompt,
tool schema, model or generation parameter is changed. END always waits for the
full response and the existing premature-END check. Any invalid/truncated stream
still fails the episode, even if generation already began speculatively.
"""
from concurrent.futures import Future
import re
import threading
import time
from types import SimpleNamespace

from worldsync.reference_player import ACTIONS, OPERATIONS, decide


def action_prefix(text):
    """Accept complete quoted enum fields, never a partial action string."""
    fields = {}
    # Only inspect the object prefix before reason, whose free text may contain
    # escaped words that look like tool fields.
    prefix = text.split('"reason"', 1)[0]
    for name in ("operation", "action"):
        found = re.search(r'"' + name + r'"\s*:\s*"([A-Z]+)"', prefix)
        if found:
            fields[name] = found.group(1)
    if fields.get("operation") in OPERATIONS - {"END"} and fields.get("action") in ACTIONS:
        return fields
    return None


class StreamingDecision:
    def __init__(self, client, *args, controller=decide, **kwargs):
        self.started = time.monotonic()
        self.ready = threading.Event()
        self.future = Future()
        self.early = None
        self.early_at = None
        self.complete_at = None
        self.stream = None
        self.used_early = False
        self.rate_limits = {}
        owner = self

        class Messages:
            def create(self, **request):
                with client.messages.stream(**request) as stream:
                    owner.stream = stream
                    headers = getattr(getattr(stream, "response", None), "headers", {})
                    owner.rate_limits = {key: headers[key] for key in (
                        "anthropic-ratelimit-requests-limit", "anthropic-ratelimit-input-tokens-limit",
                        "anthropic-ratelimit-output-tokens-limit") if key in headers}
                    buffer, selected = "", None
                    for event in stream:
                        if (event.type == "content_block_start" and event.content_block.type == "tool_use"
                                and event.content_block.name == "choose_action"):
                            selected, buffer = event.index, ""
                        elif (event.type == "content_block_delta" and event.index == selected
                              and event.delta.type == "input_json_delta"):
                            buffer += event.delta.partial_json
                            prefix = action_prefix(buffer)
                            if prefix is not None and owner.early is None:
                                owner.early, owner.early_at = prefix, time.monotonic()
                                owner.ready.set()
                    response = stream.get_final_message()
                    owner.stream = None
                    return response

        def work():
            try:
                result = controller(SimpleNamespace(messages=Messages()), *args, **kwargs)
                self.complete_at = time.monotonic()
                self.future.set_result(result)
            except BaseException as error:
                self.future.set_exception(error)
            finally:
                self.ready.set()
        self.thread = threading.Thread(target=work, name="worldsync-player-stream", daemon=True)
        self.thread.start()

    def action(self):
        self.ready.wait()
        if self.future.done():
            return self.finish()
        if self.early is None:
            return self.finish()
        self.used_early = True
        return {**self.early, "reason": "Awaiting complete streamed explanation"}, {
            "player_transport": "stream", "player_response_pending": True}

    def finish(self):
        waited = time.monotonic()
        decision, metadata = self.future.result()
        if self.used_early and any(decision[key] != self.early[key] for key in ("operation", "action")):
            raise RuntimeError("Completed tool response differs from speculative streamed action")
        return decision, {**metadata, "player_transport": "stream", "player_response_pending": False,
            "player_rate_limits": self.rate_limits,
            "player_stream_timing": {
                "action_ready_s": (self.early_at or self.complete_at) - self.started,
                "full_response_s": self.complete_at - self.started,
                "wait_for_completion_s": time.monotonic() - waited,
                "generation_started_before_response_completed": self.used_early}}

    def close(self):
        if not self.future.done() and self.stream is not None:
            try:
                self.stream.close()
            except Exception:
                pass  # Preserve the original generation/API failure during cleanup.
        self.thread.join(timeout=5)
