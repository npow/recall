"""Workflow gate: experimental generation stays locked until reproduction is recorded."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require_reproduced_baseline(root: Path = ROOT):
    state_path = root / "reproduction/state.json"
    message = "Hillclimbing is locked: measure the frozen local stock baseline and verify its control adapter first. See reproduction/README.md."
    if not state_path.is_file():
        raise RuntimeError(message)
    state = json.loads(state_path.read_text())
    if state.get("phase") != "hillclimbing" or state.get("baseline_verified") is not True:
        raise RuntimeError(message)
    report_name = state.get("baseline_report")
    if not report_name:
        raise RuntimeError(message)
    report_path = (root / report_name).resolve()
    if not report_path.is_relative_to(root.resolve()) or not report_path.is_file():
        raise RuntimeError(message)
    report = json.loads(report_path.read_text())
    if report.get("status") != "baseline_verified" or report.get("protocol_id") != state.get("protocol_id"):
        raise RuntimeError(message)
    if not all(report.get(key) is True for key in ("coverage_verified", "protocol_verified", "control_adapter_verified", "scoring_verified")):
        raise RuntimeError(message)
