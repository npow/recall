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
