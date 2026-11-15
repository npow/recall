import argparse
import json

import pytest

from worldsync import lingbot_tuning as tuning
from worldsync.lingbot_diagnostics import summarize


def parse(argv, tmp_path, phase="reproduction", reproduced=False):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"phase": phase, "baseline_reproduced": reproduced}))
    parser = argparse.ArgumentParser()
    tuning.add_arguments(parser)
    return tuning.resolve(parser.parse_args(argv), parser, state)


def test_stock_defaults_always_allowed(tmp_path):
    assert tuning.is_stock(parse([], tmp_path))


def test_non_stock_knobs_locked_until_gate_opens(tmp_path):
    with pytest.raises(SystemExit):
        parse(["--sink-size", "2"], tmp_path)
    knobs = parse(["--sink-size", "2", "--decision-interval", "2"], tmp_path, "hillclimbing", True)
    assert knobs["sink_size"] == 2 and knobs["decision_interval"] == 2


def test_invalid_cache_rejected(tmp_path):
    with pytest.raises(SystemExit):
        parse(["--local-attn-size", "6"], tmp_path, "hillclimbing", True)


def test_actions_held_between_decisions():
    history = [{"operation": "KEEP", "action": "W"}]
    assert tuning.held_decision(history, 1, 1) is None
    assert tuning.held_decision(history, 1, 3)["action"] == "W"
    assert tuning.held_decision(history, 3, 3) is None
    assert tuning.held_decision([], 1, 3) is None


def test_diagnostics_detect_turn_and_revisit():
    # One RIGHT chunk is a 60 degree turn at the 60 deg/s mapping.
    report = summarize(["W"] * 5 + ["RIGHT"] * 3 + ["W"] * 5)
    assert report["net_yaw_degrees"] == pytest.approx(180)
    assert report["revisited_start_at_chunk"] is not None
    ninety = summarize(["W"] * 5 + ["RIGHT"] * 2 + ["W"] * 5)
    assert ninety["net_yaw_degrees"] == pytest.approx(120)
    assert ninety["revisited_start_at_chunk"] is None


def test_wait_holds_pose():
    report = summarize(["W", "W", "WAIT", "WAIT", "LEFT"])
    assert report["wait_chunks"] == 2 and report["wait_pose_changes"] == 0
