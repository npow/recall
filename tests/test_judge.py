import pytest

from worldsync.judge import effective_score, parse_json, metric_task_record


def test_failed_trajectory_scores_one_even_if_appearance_is_perfect():
    assert effective_score({"verdict": "fail"}, {"final_score_1_to_5": 5}) == 1
    assert effective_score({"verdict": "fail"}, None) == 1


def test_valid_trajectory_uses_rubric_score():
    assert effective_score({"verdict": "pass"}, {"final_score_1_to_5": 2.5}) == 2.5
    with pytest.raises(ValueError):
        effective_score({"verdict": "pass"}, {"final_score_1_to_5": True})


def test_parse_json_accepts_fences_and_rejects_nonobjects():
    assert parse_json('```json\n{"verdict":"fail"}\n```') == {"verdict": "fail"}
    with pytest.raises(ValueError):
        parse_json("[]")


def test_dataset_perspectives_dispatch_to_released_gates():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "external/PlayWorld/metrics/vqa/instruction_gate.py"
    if not path.exists():
        pytest.skip("pinned PlayWorld checkout not available")
    spec = importlib.util.spec_from_file_location("released_gate", path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    original = {"task_id": "IF001", "perspective": "third person", "questions": ["unchanged"]}
    converted = metric_task_record(original)
    assert gate.gate_kind(converted) == "if_third_person"
    assert original["perspective"] == "third person"
    assert converted["questions"] == original["questions"]
    assert gate.gate_kind(metric_task_record({"task_id": "OE001", "sub_category": "insight evolution"})) is None
    assert gate.gate_kind(metric_task_record({"task_id": "OE014", "sub_category": "outofsight evolution"})) == "oe_out_of_sight"
