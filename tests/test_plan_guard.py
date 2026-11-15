from worldsync.plan_guard import end_deviation, expected_state, parse_steps


def test_parse_steps():
    assert parse_steps(["hold(W,7200ms)", "wait(900ms)", "hold(left,1800ms)"]) == [("W", 7200), ("WAIT", 900), ("LEFT", 1800)]


def test_out_and_back_expects_origin():
    state = expected_state(["hold(W,7200ms)", "hold(S,7200ms)"])
    assert abs(state["position_normalized_units"][2]) < 1e-9 and state["path_length"] > 0


def test_rejects_turned_around_after_translation_only_plan():
    plan = ["hold(W,7200ms)", "hold(S,7200ms)"]
    assert end_deviation({"heading_degrees": 180, "position_normalized_units": [0, 0, 1.0]}, plan)


def test_rejects_far_from_start_when_plan_returns():
    plan = ["hold(W,7200ms)", "hold(S,7200ms)"]
    assert end_deviation({"heading_degrees": 0, "position_normalized_units": [0, 0, 12.0]}, plan)


def test_accepts_clean_return_and_never_enforces_rotation_plans():
    plan = ["hold(W,7200ms)", "hold(S,7200ms)"]
    assert end_deviation({"heading_degrees": 5, "position_normalized_units": [0, 0, 1.0]}, plan) is None
    assert end_deviation({"heading_degrees": 45, "position_normalized_units": [0, 0, 0]}, ["hold(Right,9450ms)"]) is None
    assert end_deviation({"heading_degrees": 0, "position_normalized_units": [0, 0, 0]}, None) is None
