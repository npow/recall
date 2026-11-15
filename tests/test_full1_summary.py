import json
from pathlib import Path

import pytest

from scripts.summarize_full1 import summarize

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def matrix(tmp_path):
    plan = json.loads((ROOT / 'reproduction/full1-recovered-judging/generation-plan.json').read_text())
    (tmp_path / 'generation-plan.json').write_text(json.dumps(plan))
    return tmp_path, plan


def test_missing_cases_produce_bounds_and_withhold_exact_overall(matrix):
    path, _ = matrix
    result = summarize(path)
    assert result['complete'] is False
    for arm in result['arms'].values():
        assert arm['overall_equal_dimension_mean'] is None
        assert arm['possible_overall_range'] == [1, 5]


def test_duplicate_case_cannot_replace_missing_case_at_same_split_count(matrix):
    path, plan = matrix
    indices = [i for i, case in enumerate(plan['cases']) if case.startswith('gc:')]
    plan['cases'][indices[0]] = plan['cases'][indices[1]]
    (path / 'generation-plan.json').write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='duplicate'):
        summarize(path)


def test_wrong_task_membership_is_rejected_even_with_correct_counts(matrix):
    path, plan = matrix
    index = next(i for i, case in enumerate(plan['cases']) if case.startswith('gc:'))
    plan['cases'][index] = 'gc:GC999'
    (path / 'generation-plan.json').write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='membership'):
        summarize(path)


def test_boolean_score_cannot_count_as_valid_one(matrix):
    path, plan = matrix
    split, task = plan['cases'][0].split(':')
    identifier = f'stock-{split}-{task}-seed42'
    (path / 'episodes').mkdir()
    (path / 'episodes' / (identifier+'.json')).write_text(json.dumps(
        {'episode_id': identifier, 'task_id': task, 'scores': {'official': True}}))
    with pytest.raises(ValueError, match='Invalid score'):
        summarize(path)
