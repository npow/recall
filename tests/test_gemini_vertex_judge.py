import json
from pathlib import Path
import shutil
import subprocess

import pytest

from worldsync import gemini_vertex_judge as gvj

PLAYWORLD = Path(__file__).resolve().parents[1] / "external/PlayWorld"
needs_playworld = pytest.mark.skipif(not (PLAYWORLD / "data/gc/data.json").is_file(), reason="PlayWorld data not available")


def test_gcloud_credentials_refresh_uses_gcloud_and_sets_expiry():
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        return "ya29.fake-token\n"

    credentials = gvj.gcloud_credentials_class()(account="me@example.com", runner=runner, clock=lambda: 1000.0)
    credentials.refresh(None)
    assert credentials.token == "ya29.fake-token"
    assert calls == [["gcloud", "auth", "print-access-token", "--account=me@example.com"]]
    assert credentials.expiry.timestamp() > 0
    assert not credentials.valid or credentials.expiry is not None


def test_gcloud_credentials_reject_empty_token():
    credentials = gvj.gcloud_credentials_class()(runner=lambda *a, **k: "  \n")
    with pytest.raises(RuntimeError):
        credentials.refresh(None)


def test_resolve_project_prefers_argument_then_gcloud(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCLOUD_PROJECT", raising=False)
    assert gvj.resolve_project("explicit") == "explicit"
    assert gvj.resolve_project(runner=lambda *a, **k: "from-gcloud\n") == "from-gcloud"
    with pytest.raises(RuntimeError):
        gvj.resolve_project(runner=lambda *a, **k: "\n")


class FlakyClient:
    def __init__(self, failures, error="429 RESOURCE_EXHAUSTED"):
        self.failures, self.error, self.calls = failures, error, 0
        self.models = self

    def generate_content(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError(self.error)
        return "ok"


def test_retries_back_off_then_succeed():
    sleeps = []
    client = FlakyClient(2)
    assert gvj.generate_with_retries(client, "m", "c", None, sleep=sleeps.append) == "ok"
    assert client.calls == 3 and sleeps == [4.0, 8.0]


def test_non_retryable_error_raises_immediately():
    client = FlakyClient(5, error="400 INVALID_ARGUMENT bad request")
    with pytest.raises(RuntimeError):
        gvj.generate_with_retries(client, "m", "c", None, sleep=lambda s: None)
    assert client.calls == 1


def test_retry_budget_is_bounded():
    client = FlakyClient(99)
    with pytest.raises(RuntimeError):
        gvj.generate_with_retries(client, "m", "c", None, retries=2, sleep=lambda s: None)
    assert client.calls == 3


def test_cache_key_depends_on_every_input(tmp_path):
    base = gvj.cache_path(tmp_path, "v", "GC001", "gate", "m", "p")
    assert base == gvj.cache_path(tmp_path, "v", "GC001", "gate", "m", "p")
    for changed in [("v2", "GC001", "gate", "m", "p"), ("v", "GC002", "gate", "m", "p"), ("v", "GC001", "score", "m", "p"),
                    ("v", "GC001", "gate", "m2", "p"), ("v", "GC001", "gate", "m", "p2")]:
        assert gvj.cache_path(tmp_path, *changed) != base


def test_perspective_bridge_keeps_other_fields():
    task = {"task_id": "GC001", "source_task_id": "GC001", "perspective": "first person", "prompt": "x"}
    bridged = gvj.metric_task_record(task)
    assert bridged["perspective"] == "first-person" and bridged["prompt"] == "x"


def test_unwrap_json_object_accepts_one_element_list_only():
    assert gvj.unwrap_json_object([{"verdict": "pass"}]) == {"verdict": "pass"}
    assert gvj.unwrap_json_object({"verdict": "fail"}) == {"verdict": "fail"}
    for bad in ([], [{"a": 1}, {"b": 2}], "text", [1]):
        with pytest.raises(ValueError):
            gvj.unwrap_json_object(bad)


@needs_playworld
def test_failed_gate_scores_one_without_rubric():
    gvj.load_playworld(PLAYWORLD)
    assert gvj.effective_scores({"verdict": "fail"}, None) == {"official": 1.0, "final_score": 1.0}
    assert gvj.effective_scores({"verdict": "pass"}, None)["official"] is None


@needs_playworld
def test_gc_official_score_weights_identity_double():
    gvj.load_playworld(PLAYWORLD)
    rubric = {"task_id": "GC001", "final_score_1_to_5": 3.0,
              "categories": {"identity_id": {"score": 4, "weight": 1}, "other": {"score": 1, "weight": 1}}}
    scores = gvj.effective_scores({"verdict": "partial"}, rubric)
    assert scores["official"] == round((4 * 2 + 1 * 1) / 3, 2)
    assert scores["final_score"] == 3.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None or not (PLAYWORLD / "data/gc/data.json").is_file(), reason="needs ffmpeg and PlayWorld data")
def test_dry_run_samples_video_and_builds_prompts_without_api(tmp_path):
    video = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=832x480:rate=16:duration=3",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    task = gvj.load_dataset_task(PLAYWORLD, "gc", "GC007")
    result = gvj.judge_episode(None, PLAYWORLD, task, video, dry_run=True, cache_dir=tmp_path / "cache")
    assert result["dry_run"] and result["gate_prompt_sha256"] and result["score_prompt_sha256"]
    assert result["sampling"]["primary"]["fps"] == 10
    assert "scores" not in result


@pytest.mark.skipif(shutil.which("ffmpeg") is None or not (PLAYWORLD / "data/gc/data.json").is_file(), reason="needs ffmpeg and PlayWorld data")
def test_manifest_dry_run_records_each_episode(tmp_path, capsys):
    video = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=832x480:rate=16:duration=2",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"episode_id": "e1", "split": "gc", "task_id": "GC007", "video": str(video)}) + "\n")
    summary = gvj.run_manifest(manifest, tmp_path / "out", PLAYWORLD, dry_run=True, workers=1)
    assert summary[0]["episode_id"] == "e1" and summary[0].get("error") is None


def test_blank_judge_response_preserves_provider_provenance():
    from types import SimpleNamespace
    response = SimpleNamespace(text=None, response_id="provider-response", model_version="model-version",
                               candidates=[SimpleNamespace(finish_reason="STOP")], prompt_feedback=None)
    with pytest.raises(ValueError, match="Judge returned no text") as error:
        gvj.response_text(response)
    assert "provider-response" in str(error.value) and "model-version" in str(error.value)
    assert "STOP" in str(error.value)


def test_valid_judge_response_text_is_unchanged():
    from types import SimpleNamespace
    value = ' {"score": 2} \n'
    assert gvj.response_text(SimpleNamespace(text=value)) == value
