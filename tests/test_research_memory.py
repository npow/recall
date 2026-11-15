"""Mocked-HTTP tests for the past.dev research-memory client (no network, no key)."""
import io
import json
import urllib.error

import pytest

from worldsync import research_memory as rm


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, int):
            raise urllib.error.HTTPError(request.full_url, reply, "err", {}, io.BytesIO(b'{"code":"x"}'))
        return FakeResponse(json.dumps(reply).encode())


def client(replies):
    opener = FakeOpener(replies)
    return rm.PastClient(api_key="k-secret-value", opener=opener, sleep=lambda s: None), opener


def test_ingest_request_shape_and_auth():
    c, op = client([{"ingestionId": "i1", "status": "queued"}])
    finding = rm.build_finding("Memory v2 scored -0.35.", "2026-11-15T09:00:00-08:00", "a/b.md", ("memory",), section="S")
    c.ingest(finding)
    req = op.requests[0]
    body = json.loads(req.data)
    assert req.full_url.endswith("/api/v1/ingest") and req.get_method() == "POST"
    assert req.get_header("Authorization") == "Bearer k-secret-value"
    assert {"id", "content", "label", "timestamp"} <= set(body)
    assert "/" not in body["id"]
    assert "k-secret-value" not in repr(c)


def test_recall_request_shape():
    c, op = client([{"results": []}])
    c.recall("what?", k=3)
    body = json.loads(op.requests[0].data)
    assert op.requests[0].full_url.endswith("/api/v1/recall")
    assert body["query"] == "what?" and body["identity"] and body["limit"] == 3


def test_retry_then_success_and_no_retry_on_credits():
    c, op = client([503, {"ok": 1}])
    assert c._call("GET", "/x") == {"ok": 1} and len(op.requests) == 2
    c, op = client([400])
    with pytest.raises(rm.PastError):
        c._call("GET", "/x")
    assert len(op.requests) == 1


def test_redaction():
    text = "see /home/someone/code/x/reproduction/a.md, mail a.b@example.com at 10.1.2.3 key AIzaSyAAAAAAAAAAAAAAAAAAAAAAAA"
    out = rm.redact(text)
    for bad in ("/home/", "@example.com", "10.1.2.3", "AIza"):
        assert bad not in out


def test_ingest_repo_is_idempotent(tmp_path):
    findings = [rm.build_finding(f"finding {i} with numbers 1.8{i}", "2026-11-15T09:00:00-08:00", "f.md", section=f"s{i}")
                for i in range(3)]
    manifest = tmp_path / "m.jsonl"
    c, op = client([{"ingestionId": f"i{i}", "status": "queued"} for i in range(3)])
    first = rm.ingest_repo(client=c, dry_run=False, findings=findings, manifest_path=manifest, log=lambda *_: None)
    assert first["sent"] == 3
    c2, op2 = client([])
    second = rm.ingest_repo(client=c2, dry_run=False, findings=findings, manifest_path=manifest, log=lambda *_: None)
    assert second["sent"] == 0 and second["skipped"] == 3 and not op2.requests
    assert len(manifest.read_text().splitlines()) == 3


def test_missing_key_refused(monkeypatch):
    monkeypatch.delenv("PAST_API_KEY", raising=False)
    with pytest.raises(rm.PastError):
        rm.PastClient()
