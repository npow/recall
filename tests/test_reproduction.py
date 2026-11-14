import json

import pytest

from worldsync.reproduction import require_reproduced_baseline


def write_state(root, **changes):
    path = root / "reproduction/state.json"
    path.parent.mkdir(exist_ok=True)
    state = {"phase": "reproduction", "baseline_reproduced": False, "protocol_id": "test", "baseline_report": None}
    state.update(changes)
    path.write_text(json.dumps(state))


def test_reproduction_gate_rejects_missing_or_unverified_baseline(tmp_path):
    with pytest.raises(RuntimeError, match="locked"):
        require_reproduced_baseline(tmp_path)
    write_state(tmp_path)
    with pytest.raises(RuntimeError, match="locked"):
        require_reproduced_baseline(tmp_path)
    write_state(tmp_path, phase="hillclimbing", baseline_reproduced=True, baseline_report="missing.json")
    with pytest.raises(RuntimeError, match="locked"):
        require_reproduced_baseline(tmp_path)


def test_reproduction_gate_requires_protocol_coverage_and_agreement(tmp_path):
    write_state(tmp_path, phase="hillclimbing", baseline_reproduced=True, baseline_report="reproduction/verified.json")
    report = {"status": "reproduced", "protocol_id": "test", "coverage_verified": True, "protocol_verified": True, "score_agreement_verified": False}
    path = tmp_path / "reproduction/verified.json"
    path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="locked"):
        require_reproduced_baseline(tmp_path)
    report["score_agreement_verified"] = True
    path.write_text(json.dumps(report))
    require_reproduced_baseline(tmp_path)
    report["protocol_id"] = "different"
    path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="locked"):
        require_reproduced_baseline(tmp_path)
