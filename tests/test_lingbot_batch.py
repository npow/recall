import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from worldsync.lingbot_batch_rollout import groups, run_batch
from worldsync.lingbot_pipeline_reuse import PipelineReuse


def job(identifier, cache=18):
    return {"id": identifier, "argv": ["--split", "gc", "--task-id", identifier,
        "--native-fixed-controls", "--fixed-actions", "WAIT", "--offload-text-encoder",
        "--local-attn-size", str(cache)]}


def test_grouping_preserves_all_jobs_and_splits_model_cache_configuration():
    result = groups([job("one"), job("two", 22), job("three")])
    assert [[row["id"] for row in group["jobs"]] for group in result] == [["one", "three"], ["two"]]
    with pytest.raises(ValueError, match="unique"):
        groups([job("same"), job("same")])
    with pytest.raises(ValueError):
        groups([job("../escape")])
    invalid = job("adaptive")
    invalid["argv"].remove("--native-fixed-controls")
    with pytest.raises(ValueError):
        groups([invalid])
    abbreviated = job("abbreviated")
    abbreviated["argv"][abbreviated["argv"].index("--local-attn-size")] = "--local-attn-s"
    with pytest.raises(ValueError, match="Spell model"):
        groups([abbreviated])


def fake_rollout():
    current = {}

    class Pipeline:
        def __init__(self, *args, **kwargs):
            current["report"]["checkpoint_loading"] = {"missing_keys": []}
            self.model = torch.nn.Linear(2, 2)
            self.text_encoder = SimpleNamespace(model=torch.nn.Linear(2, 2))
            self._t5_cache = {"old": "context"}
            self.vae = SimpleNamespace(model=SimpleNamespace(clear_cache=lambda: None))

        def clear_text_cache(self):
            self._t5_cache.clear()

    upstream = SimpleNamespace(WanI2VCausal=Pipeline)

    def backend(report, save):
        current["report"] = report
        return upstream

    def main():
        directory = Path(sys.argv[sys.argv.index("--output") + 1])
        name = sys.argv[sys.argv.index("--task-id") + 1]
        report = {}
        configured = rollout.configure_backend(report, lambda: None)
        pipeline = configured.WanI2VCausal("config", "checkpoint")
        if name == "fail":
            raise RuntimeError("deliberate request failure")
        report.update(status="completed", max_chunks=1, native_fixed_controls=True,
                      camera_embedding_audit=[{"within_tolerance": True}],
                      chunks=[{"elapsed_s": 1., "pixel_sha256": "sample"}],
                      hardware={"gpu": "fake"}, peak_allocated_bytes=0)
        directory.mkdir(parents=True)
        (directory / "run.json").write_text(json.dumps(report))

    rollout = SimpleNamespace(configure_backend=backend, main=main)
    return rollout, upstream


def test_reuse_returns_same_pipeline_clears_prompt_cache_and_restores_constructor():
    rollout, upstream = fake_rollout()
    original = upstream.WanI2VCausal
    with PipelineReuse(rollout) as worker:
        first = worker.configure({}, lambda: None).WanI2VCausal("same", "checkpoint")
        second = worker.configure({}, lambda: None).WanI2VCausal("same", "checkpoint")
        assert first is second and not second._t5_cache
        assert worker.constructor_calls == 1
        with pytest.raises(ValueError, match="Different pipeline"):
            worker.configure({}, lambda: None).WanI2VCausal("different", "checkpoint")
        handle = first.model.register_forward_hook(lambda *_: None)
        with pytest.raises(ValueError, match="Unsafe persistent"):
            worker.audit()
        handle.remove()
        with torch.no_grad():
            first.model.weight.add_(1)
        with pytest.raises(ValueError, match="Unsafe persistent"):
            worker.audit()
    assert upstream.WanI2VCausal is original


def test_batch_retains_failed_job_and_reloads_before_next_request(monkeypatch, tmp_path):
    rollout, _ = fake_rollout()
    monkeypatch.setattr(torch, "__version__", "2.6.0+cu124")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    summary = run_batch({"jobs": [job("fail"), job("good"), job("also-good")]}, tmp_path, rollout)
    assert [row["status"] for row in summary["jobs"]] == ["failed", "completed", "completed"]
    assert "deliberate request failure" in summary["jobs"][0]["error"]
    assert summary["constructor_calls"] == 2  # Failed model discarded; successful requests share its replacement.
    assert summary["status"] == "failed"
    assert (tmp_path / "good/run.json").exists()
    assert json.loads((tmp_path / "summary.json").read_text())["declared_jobs"] == 3
