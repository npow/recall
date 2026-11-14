import json

import pytest

from worldsync.reproduction import require_reproduced_baseline


def write_state(root, **changes):
    path = root / "reproduction/state.json"
    path.parent.mkdir(exist_ok=True)
    state = {"phase": "baseline", "baseline_verified": False, "protocol_id": "test", "baseline_report": None}
    state.update(changes)
    path.write_text(json.dumps(state))


def test_reproduction_gate_rejects_missing_or_unverified_baseline(tmp_path):
    with pytest.raises(RuntimeError, match="locked"):
        require_reproduced_baseline(tmp_path)
    write_state(tmp_path)
    with pytest.raises(RuntimeError, match="locked"):
        require_reproduced_baseline(tmp_path)
    write_state(tmp_path, phase="hillclimbing", baseline_verified=True, baseline_report="missing.json")
    with pytest.raises(RuntimeError, match="locked"):
        require_reproduced_baseline(tmp_path)


def test_local_baseline_requires_controls_and_scoring_but_not_hosted_score_agreement(tmp_path):
    write_state(tmp_path, phase="hillclimbing", baseline_verified=True, baseline_report="reproduction/verified.json")
    report = {"status": "baseline_verified", "protocol_id": "test", "coverage_verified": True,
              "protocol_verified": True, "control_adapter_verified": False, "scoring_verified": True,
              "hosted_score_agreement": False}
    path = tmp_path / "reproduction/verified.json"
    path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="locked"):
        require_reproduced_baseline(tmp_path)
    report["control_adapter_verified"] = True
    path.write_text(json.dumps(report))
    require_reproduced_baseline(tmp_path)
    report["protocol_id"] = "different"
    path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="locked"):
        require_reproduced_baseline(tmp_path)
