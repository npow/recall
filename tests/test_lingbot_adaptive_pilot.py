import json
from pathlib import Path
import subprocess
import sys

import pytest

from worldsync.lingbot_adaptive_pilot import player_state
from worldsync.lingbot_controls import CameraPlan


def test_neutral_bootstrap_then_twelve_turns_closes360_and_provides_old_memory():
    from worldsync.lingbot_retrieval import select_frames
    camera = CameraPlan(turn_degrees_per_second=30)
    camera.advance("WAIT")
    for _ in range(12):
        camera.advance("RIGHT")
    state = player_state(camera)
    assert state["accumulated_yaw_degrees"] == pytest.approx(360)
    assert state["heading_degrees"] == pytest.approx(0, abs=1e-10)
    assert state["turn_degrees_per_second"] == 30
    selected = select_frames(camera.poses, range(6, 22), current_start=48)
    assert len(selected) == 4 and all(index < 44 for index in selected)


@pytest.mark.parametrize("extra,valid", [([], True), (["--pose-retrieval"], True),
    (["--kv-retention"], False), (["--task-id", "GC040"], False),
    (["--native-fixed-controls"], False), (["--local-attn-size", "22"], False),
    (["--denoise-schedule", "cmbench"], False)])
def test_pilot_is_explicit_and_cannot_unlock_other_diagnostics(tmp_path, extra, valid):
    root = Path(__file__).resolve().parents[1]
    if not (root / "external/LingBot-World2/.git").exists():
        pytest.skip("native source checkout not installed")
    output = tmp_path / "trial"
    command = [sys.executable, "-m", "worldsync.lingbot_rollout", "--split", "gc", "--task-id", "GC027",
               "--adaptive-retrieval-pilot", "--turn-degrees-per-second", "30", "--offload-text-encoder",
               "--output", str(output), "--dry-run", *extra]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    if valid:
        assert result.returncode == 0, result.stderr
        report = json.loads((output / "run.json").read_text())
        assert report["protocol_id"] == "lingbot-world2-adaptive-four-view-pilot-v1"
        assert report["initial_neutral_bootstrap"] and report["player_temperature"] == "provider default"
        assert not report["native_fixed_controls"] and report["agent_interactions"] == 0
    else:
        assert result.returncode == 2
        assert not output.exists()
