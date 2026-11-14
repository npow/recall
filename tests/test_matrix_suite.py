import json
from types import SimpleNamespace

import pytest

from worldsync import matrix_suite


def test_suite_resumes_completed_cases_and_stops_without_dropping_failure(tmp_path, monkeypatch):
    cases = [{"split": "gc", "task_id": "GC007"}, {"split": "if", "task_id": "IF001"}]
    monkeypatch.setattr(matrix_suite, "inventory_cases", lambda root: cases.copy())
    monkeypatch.setattr("sys.argv", ["matrix_suite", "--output", str(tmp_path), "--max-chunks", "1"])
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        directory = tmp_path / ("gc-GC007-attempt1" if len(calls) == 1 else "if-IF001-attempt1")
        directory.mkdir()
        if len(calls) == 1:
            (directory / "run.json").write_text(json.dumps({"status": "completed", "generated_frames": 57,
                "generated_seconds": 57 / 17, "chunks": [{}], "termination": "chunk_budget_exhausted"}))
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(matrix_suite.subprocess, "run", execute)
    with pytest.raises(SystemExit, match="No cases silently dropped"):
        matrix_suite.main()
    manifest = json.loads((tmp_path / "suite.json").read_text())
    assert manifest["completed_by_split"] == {"gc": 1, "if": 0, "insight": 0, "outsight": 0}
    assert manifest["status"] == "stopped_on_failure"
    assert manifest["published_score_reproduced"] is False
    def recover(command, **kwargs):
        assert command[command.index("--task-id") + 1] == "IF001"
        directory = tmp_path / "if-IF001-attempt2"
        directory.mkdir()
        (directory / "run.json").write_text(json.dumps({"status": "completed", "chunks": [{}]}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(matrix_suite.subprocess, "run", recover)
    matrix_suite.main()
    manifest = json.loads((tmp_path / "suite.json").read_text())
    assert manifest["status"] == "generation_complete_scoring_unverified"
    assert len(manifest["runs"]) == 3
    assert manifest["runs"][1]["status"] == "failed"
    assert manifest["completed_by_split"]["gc"] == 1
    assert manifest["completed_by_split"]["if"] == 1


def test_judge_retry_reuses_successful_generation(tmp_path, monkeypatch):
    cases = [{"split": "gc", "task_id": "GC007"}]
    monkeypatch.setattr(matrix_suite, "inventory_cases", lambda root: cases.copy())
    data = tmp_path / "data/gc"
    data.mkdir(parents=True)
    (data / "data.json").write_text(json.dumps([{"task_id": "GC007", "image_path": "gc/initial.jpg"}]))
    output = tmp_path / "output"
    monkeypatch.setattr("sys.argv", ["matrix_suite", "--output", str(output), "--data-root", str(data.parent), "--judge"])
    calls = {"generation": 0, "judge": 0}
    def execute(command, **kwargs):
        directory = output / "gc-GC007-attempt1"
        if "worldsync.matrix_baseline" in command:
            calls["generation"] += 1
            directory.mkdir()
            (directory / "run.json").write_text(json.dumps({"status": "completed", "chunks": [{}]}))
            return SimpleNamespace(returncode=0)
        calls["judge"] += 1
        if calls["judge"] == 1:
            (directory / "judge.json").write_text(json.dumps({"reported_score": None}))
            return SimpleNamespace(returncode=1)
        (directory / "judge.json").write_text(json.dumps({"reported_score": 1, "gate": {"verdict": "fail"}}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(matrix_suite.subprocess, "run", execute)
    with pytest.raises(SystemExit, match="Judge failed"):
        matrix_suite.main()
    matrix_suite.main()
    assert calls == {"generation": 1, "judge": 2}
    manifest = json.loads((output / "suite.json").read_text())
    assert manifest["runs"][0]["diagnostic_score"] == 1
    assert manifest["diagnostic_scores"]["diagnostic_overall"] is None
    assert (output / "gc-GC007-attempt1/judge-prior-1.json").is_file()
