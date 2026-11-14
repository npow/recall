import pytest

from worldsync.judge import effective_score, parse_json


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
