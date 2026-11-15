import json

from worldsync import fourdim


def _data(tmp_path, counts):
    for split, count in counts.items():
        directory = tmp_path / split
        directory.mkdir()
        (directory / "data.json").write_text(json.dumps([{"task_id": f"{split.upper()}{i:03d}"} for i in range(count)]))


def test_selection_is_deterministic_stratified_and_outcome_blind(tmp_path):
    _data(tmp_path, {"gc": 20, "if": 20, "insight": 15, "outsight": 14})
    first = fourdim.select_cases(tmp_path, 12)
    assert first == fourdim.select_cases(tmp_path, 12)
    assert [sum(1 for s, _ in first if s == split) for split in fourdim.SPLITS] == [12, 12, 12, 12]
    assert len(set(first)) == 48
    assert first != fourdim.select_cases(tmp_path, 12, salt="other")


def test_overall_is_equal_mean_of_dimension_means():
    scores = {("gc", "a"): 2.0, ("gc", "b"): 4.0, ("if", "a"): 1.0, ("insight", "a"): 1.0, ("outsight", "a"): 1.0}
    dims = fourdim.dimension_means(scores)
    assert dims["gc"] == 3.0
    assert fourdim.overall(dims) == (3.0 + 1.0 + 1.0 + 1.0) / 4
    assert fourdim.overall({**dims, "if": None}) is None


def test_paired_bootstrap_matches_known_delta():
    a = {(s, str(i)): 3.0 for s in fourdim.SPLITS for i in range(5)}
    b = {key: 2.0 for key in a}
    result = fourdim.bootstrap_paired_delta(a, b, n=200)
    assert result["delta"] == 1.0 and result["ci95"] == [1.0, 1.0]
    assert fourdim.bootstrap_overall(a, n=200) == [3.0, 3.0]
