import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from worldsync import lingbot_nanokv_episode as episode


def spec(**overrides):
    return {
        "id": "example",
        "split": "gc",
        "task_id": "GC007",
        "seed": 42,
        "max_chunks": 2,
        "capacity_frames": 58,
        "fixed_actions": ["WAIT", "RIGHT"],
        **overrides,
    }


class FakePipeline:
    constructor_calls = 0

    def __init__(self, **kwargs):
        type(self).constructor_calls += 1
        self.model = torch.nn.Linear(1, 1).eval()
        self.text_encoder = SimpleNamespace(model=torch.nn.Linear(1, 1))
        self.vae = SimpleNamespace(model=SimpleNamespace(clear_cache=lambda: None))
        self._t5_cache = {}

    def clear_text_cache(self):
        self._t5_cache.clear()


@pytest.fixture
def harness(monkeypatch):
    from worldsync import lingbot_nanokv, lingbot_rollout

    created, pipelines, calls = [], [], []
    upstream = SimpleNamespace(WanI2VCausal=FakePipeline)
    monkeypatch.setattr(
        lingbot_rollout, "configure_backend", lambda report, save: upstream
    )

    def build(*args, **kwargs):
        module = lingbot_rollout.configure_backend(
            {"checkpoint_loading": {}}, lambda: None
        )
        pipe = module.WanI2VCausal(capacity=58)
        pipelines.append(pipe)
        return pipe

    class Adapter:
        def __init__(self, pipe, **kwargs):
            self.pipe, self.kwargs, self.closed = pipe, kwargs, False
            self.handle = None
            created.append(self)

        def __enter__(self):
            self.handle = self.pipe.model.register_forward_pre_hook(
                lambda module, args: None
            )
            return self

        def close(self):
            if self.handle is not None:
                self.handle.remove()
                self.handle = None
            self.closed = True

        def metadata(self):
            return {
                "closed": self.closed,
                "identity": False,
                "capacity_frames": self.kwargs["capacity_frames"],
            }

    def main(argv):
        calls.append(argv)
        lingbot_rollout.build_pipeline()
        destination = Path(argv[argv.index("--output") + 1])
        destination.mkdir(parents=True)
        report = {
            "status": "completed",
            "chunks": [{"pixel_sha256": "a"}, {"pixel_sha256": "b"}],
            "camera_embedding_audit": [
                {"within_tolerance": True},
                {"within_tolerance": True},
            ],
        }
        (destination / "run.json").write_text(json.dumps(report))
        return report

    monkeypatch.setattr(lingbot_rollout, "build_pipeline", build)
    monkeypatch.setattr(lingbot_rollout, "main", main)
    monkeypatch.setattr(lingbot_nanokv, "NativeNanoKV", Adapter)
    return SimpleNamespace(
        rollout=lingbot_rollout,
        created=created,
        pipelines=pipelines,
        calls=calls,
        build=build,
        main=main,
    )


def test_fixed_job_annotations_and_full_cleanup(tmp_path, harness):
    summary = episode.run_episode(
        spec(), tmp_path / "job", codec_path=Path("codec"), source_root=Path("source")
    )
    assert summary["status"] == "completed"
    assert (
        summary["base_parameter_versions_unchanged"] and summary["model_hooks_restored"]
    )
    assert summary["all_native_camera_audits_passed"]
    assert summary["raw_pixel_hashes"] == ["a", "b"]
    assert (
        "--native-fixed-controls" in harness.calls[0]
        and "--controlled-ablation" in harness.calls[0]
    )
    assert harness.rollout.build_pipeline is harness.build and harness.created[0].closed
    assert not harness.pipelines[0].model._forward_pre_hooks
    run = json.loads((tmp_path / "job/episode/run.json").read_text())
    assert not run["stock_knobs"] and run["nanokv_episode"]["adapter"]["closed"]


def test_adaptive_argv_keeps_existing_gate_and_streaming(tmp_path, harness):
    summary = episode.run_episode(
        spec(fixed_actions=None, stream_player_actions=True),
        tmp_path / "job",
        codec_path=Path("codec"),
        source_root=Path("source"),
    )
    argv = harness.calls[0]
    assert "--stream-player-actions" in argv
    assert "--native-fixed-controls" not in argv and "--controlled-ablation" not in argv
    assert "--fixed-actions" not in argv
    assert summary["adaptive_uses_existing_hillclimb_gate"]


def test_failure_preserves_raw_run_and_closes_adapter(tmp_path, harness, monkeypatch):
    def fail(argv):
        report = harness.main(argv)
        report["status"] = "failed"
        destination = Path(argv[argv.index("--output") + 1])
        (destination / "run.json").write_text(json.dumps(report))
        raise RuntimeError("native failure")

    monkeypatch.setattr(harness.rollout, "main", fail)
    with pytest.raises(RuntimeError, match="native failure"):
        episode.run_episode(
            spec(),
            tmp_path / "job",
            codec_path=Path("codec"),
            source_root=Path("source"),
        )
    summary = json.loads((tmp_path / "job/summary.json").read_text())
    run = json.loads((tmp_path / "job/episode/run.json").read_text())
    assert summary["status"] == run["status"] == "failed"
    assert "native failure" in summary["error"] and harness.created[0].closed
    assert harness.rollout.build_pipeline is harness.build and not episode._RUNNING


@pytest.mark.parametrize("defect", ["missing_audit", "changed_parameter"])
def test_required_native_evidence_and_parameter_guard(
    tmp_path, harness, monkeypatch, defect
):
    def faulty(argv):
        report = harness.main(argv)
        if defect == "missing_audit":
            report["camera_embedding_audit"] = []
        else:
            with torch.no_grad():
                harness.pipelines[-1].model.weight.add_(1)
        return report

    monkeypatch.setattr(harness.rollout, "main", faulty)
    with pytest.raises((ValueError, RuntimeError)):
        episode.run_episode(
            spec(),
            tmp_path / "job",
            codec_path=Path("codec"),
            source_root=Path("source"),
        )
    assert harness.created[0].closed and harness.rollout.build_pipeline is harness.build
    assert json.loads((tmp_path / "job/summary.json").read_text())["status"] == "failed"


def test_serial_requests_inside_real_pipeline_reuse(tmp_path, harness):
    from worldsync.lingbot_pipeline_reuse import PipelineReuse

    FakePipeline.constructor_calls = 0
    with PipelineReuse(harness.rollout) as worker:
        for index in range(2):
            episode.run_episode(
                spec(id=str(index)),
                tmp_path / str(index),
                codec_path=Path("codec"),
                source_root=Path("source"),
            )
            assert worker.audit()["dit_parameter_versions_unchanged"]
        assert worker.constructor_calls == FakePipeline.constructor_calls == 1
        assert harness.pipelines[0] is harness.pipelines[1]
    assert all(adapter.closed for adapter in harness.created)


def checksum(plan):
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    plan["plan_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return plan


def test_plan_checks_every_input_and_all_jobs(tmp_path):
    (tmp_path / "source").write_text("frozen")
    plan = checksum(
        {
            "jobs": [spec()],
            "input_files": {"source": hashlib.sha256(b"frozen").hexdigest()},
        }
    )
    assert episode.validate_plan(plan, root=tmp_path) is plan
    changed = checksum({**plan, "jobs": [spec(), spec(id="bad", fixed_actions=[])]})
    with pytest.raises(ValueError, match="Fixed actions"):
        episode.validate_plan(changed, root=tmp_path)
    (tmp_path / "source").write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        episode.validate_plan(plan, root=tmp_path)


def test_bad_specs_and_plan_path_escape_fail_before_run(tmp_path):
    for kwargs in (
        {"capacity_frames": 131},
        {"max_chunks": True},
        {"unexpected": 1},
        {"turn_degrees_per_second": float("nan")},
        {"fixed_actions": ["WAIT"]},
    ):
        with pytest.raises(ValueError):
            episode.normalize_spec(spec(**kwargs))
    plan = checksum({"jobs": [spec()], "input_files": {"../outside": "0" * 64}})
    with pytest.raises(ValueError, match="repository"):
        episode.validate_plan(plan, root=tmp_path)


def test_cli_validates_plan_then_selects_exact_job(tmp_path, monkeypatch):
    from worldsync.lingbot_nanokv import CODEC_SHA256

    source = "worldsync/lingbot_nanokv_episode.py"
    plan = checksum(
        {
            "jobs": [spec()],
            "codec_sha256": CODEC_SHA256,
            "input_files": {source: episode._sha(episode.ROOT / source)},
        }
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    codec = tmp_path / "codec.pt"
    original_sha = episode._sha
    monkeypatch.setattr(
        episode, "_sha", lambda p: CODEC_SHA256 if Path(p) == codec else original_sha(p)
    )
    calls = []

    def run(selected, output, **kwargs):
        calls.append((selected, output, kwargs))
        return {"status": "mock-completed"}

    monkeypatch.setattr(episode, "run_episode", run)
    args = [
        "--plan",
        str(path),
        "--job-id",
        "example",
        "--output",
        str(tmp_path / "out"),
        "--codec",
        str(codec),
    ]
    assert episode.main(args) == {"status": "mock-completed"}
    assert calls[0][0] == spec() and calls[0][2]["codec_path"] == codec
    plan["jobs"][0]["seed"] = 43
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="checksum"):
        episode.main(args)
    assert len(calls) == 1
