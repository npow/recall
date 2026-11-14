import pytest

from worldsync.coverage import diagnostic_summary


def test_overall_requires_full_coverage_and_equal_dimension_weight():
    cases = [{"split": split, "task_id": f"{split}{i}"} for split, count in
             {"gc": 1, "if": 3, "insight": 1, "outsight": 1}.items() for i in range(count)]
    runs = [{**case, "status": "completed", "judge_status": "completed",
             "diagnostic_score": 5 if case["split"] == "if" else 1} for case in cases]
    expected = {"gc": 1, "if": 3, "insight": 1, "outsight": 1}
    assert diagnostic_summary(cases, runs, expected)["diagnostic_overall"] == 2
    partial = diagnostic_summary(cases, runs[:-1], expected)
    assert partial["diagnostic_overall"] is None
    assert partial["coverage_complete"] is False
    assert diagnostic_summary(cases[:-1], runs[:-1], expected)["diagnostic_overall"] is None


def test_missing_or_failed_judges_never_become_successful_scores():
    cases = [{"split": "gc", "task_id": "GC001"}]
    run = {**cases[0], "status": "failed", "judge_status": "completed", "diagnostic_score": 5}
    assert diagnostic_summary(cases, [run])["dimensions"]["gc"]["scored"] == 0
    run["status"] = "completed"
    with pytest.raises(ValueError, match="multiple scored"):
        diagnostic_summary(cases, [run, run])
