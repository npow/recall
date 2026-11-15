"""Gemini judge through Vertex AI using the machine's gcloud login (no API key).

Mirrors the released PlayWorld scorer's call: one user message of text and image
parts, JSON output, temperature 0.1, no system prompt. Tokens come from
`gcloud auth print-access-token` and are refreshed automatically.
"""
import base64
from datetime import datetime, timedelta, timezone
import subprocess
import time

DEFAULT_MODEL = "gemini-3.1-pro-preview"  # Vertex name; AI Studio calls it gemini-3.1-pro


def _gcloud(*args):
    return subprocess.check_output(["gcloud", *args], text=True, stderr=subprocess.DEVNULL).strip()


def make_client(project=None, location="global"):
    from google import genai
    from google.auth.credentials import Credentials

    class GcloudCredentials(Credentials):
        def __init__(self):
            super().__init__()
            self.token, self.expiry = None, None

        def refresh(self, request):
            self.token = _gcloud("auth", "print-access-token")
            self.expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=45)

        def before_request(self, request, method, url, headers):
            if not self.valid:
                self.refresh(request)
            headers["authorization"] = f"Bearer {self.token}"

    project = project or _gcloud("config", "get-value", "project")
    return genai.Client(vertexai=True, project=project, location=location, credentials=GcloudCredentials())


def generate(client, model, blocks, attempts=4):
    """blocks: Claude-style content blocks ({type:text|image,...}); returns (text, metadata)."""
    from google.genai import types
    parts = []
    for block in blocks:
        if block["type"] == "text":
            parts.append(types.Part.from_text(text=block["text"]))
        else:
            source = block["source"]
            parts.append(types.Part.from_bytes(data=base64.b64decode(source["data"]), mime_type=source["media_type"]))
    last = None
    for attempt in range(attempts):
        try:
            response = client.models.generate_content(
                model=model, contents=types.Content(role="user", parts=parts),
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1))
            usage = response.usage_metadata
            return response.text, {"model": model, "provider": "vertex-gemini",
                "prompt_tokens": getattr(usage, "prompt_token_count", None),
                "output_tokens": getattr(usage, "candidates_token_count", None),
                "thinking_tokens": getattr(usage, "thoughts_token_count", None)}
        except Exception as error:  # transient quota / 5xx; retry with backoff
            last = error
            time.sleep(5 * 2 ** attempt)
    raise last
