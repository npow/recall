import json

from scripts import evaluate_task_memory as evaluation


def test_invalid_source_input_never_calls_a_judge_and_keeps_raw_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    reproduction = tmp_path / "reproduction"
    reproduction.mkdir()
    (reproduction / "task-input-audit.json").write_text(json.dumps({"cases": {"BAD": {"status": "quarantined"}}}))
    directory = reproduction / "task-memory-benchmark/BAD-seed42-path"
    directory.mkdir(parents=True)
    (directory / "stock-task-score.json").write_text(json.dumps({"reported_score": 2.4}))
    monkeypatch.setattr(evaluation, "evaluate", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Invalid input reached evaluator")))
    result = evaluation.evaluate_task({"task_id": "BAD", "seed": 42, "path": "path"})
    assert result["status"] == "invalid_input"
    assert result["raw_invalid_scores_not_in_mean"] == {"stock": 2.4}


def test_partial_valid_cohort_has_no_mean_and_quarantined_case_stays_visible():
    plan = {"scope": "test", "cases": [{"task_id": key} for key in ("A", "B", "BAD")], "quarantined_case_ids": ["BAD"]}
    scored = {"task_id": "A", "status": "completed", "local_task_scores": {"stock": {"score": 2}, "retrieval": {"score": 4}}}
    invalid = {"task_id": "BAD", "status": "invalid_input", "raw_invalid_scores_not_in_mean": {"stock": 1000}}
    partial = evaluation.summarize_results(plan, [scored, invalid])
    assert not partial["complete"] and partial["local_cohort_means"] is None
    assert partial["declared_pairs_including_quarantine"] == 3
    assert partial["valid_declared_pairs"] == 2
    duplicate = evaluation.summarize_results(plan, [scored, scored, invalid])
    assert not duplicate["complete"] and duplicate["local_cohort_means"] is None
    finished = evaluation.summarize_results(plan, [scored, {**scored, "task_id": "B"}, invalid])
    assert finished["complete"]
    assert finished["local_cohort_means"] == {"stock": 2, "retrieval": 4}
    assert invalid in finished["results"]
