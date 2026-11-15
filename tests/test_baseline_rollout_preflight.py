import json

import pytest

from worldsync import lingbot_rollout, lingbot_tuning
from worldsync.lingbot_released_baseline import preflight, generation_argv, run


def test_real_rollout_preflight_stamps_controller_and_preserves_limitations(tmp_path, monkeypatch):
    monkeypatch.setattr(lingbot_tuning, 'hillclimb_unlocked', lambda: False)
    profile = preflight('gc', 'GC010')
    report = lingbot_rollout.main(generation_argv('gc', 'GC010', tmp_path, chunks=2)+['--dry-run'],
        player_decide=lambda *a, **k: pytest.fail('No player request during preflight'),
        simulation_chunk_budget=2, controller_profile=profile)
    saved = json.loads((tmp_path/'run.json').read_text())
    assert saved == report
    assert saved['protocol_id'] == profile['profile']
    assert saved['deviations'] == profile['remaining_differences']
    assert saved['player_sha256'] == profile['source_sha256']['worldsync/released_reference_player.py']
    assert saved['observation_encoding'].startswith('lossless PNG')
    assert saved['published_score_reproduced'] is False


def test_runtime_smoke_cannot_be_used_to_launch_a_full_episode(tmp_path):
    with pytest.raises(ValueError, match='at most2chunks'):
        run('gc', 'GC010', tmp_path/'smoke', chunks=20, validate_runtime=True)


def test_runtime_failure_is_recorded_in_outer_manifest(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('simulated model load failure')
    monkeypatch.setattr(lingbot_rollout, 'main', fail)
    with pytest.raises(RuntimeError, match='model load failure'):
        run('gc', 'GC010', tmp_path/'smoke', chunks=2, validate_runtime=True)
    audit = json.loads((tmp_path/'smoke/protocol.json').read_text())
    assert audit['generation_status'] == 'failed'
    assert audit['scope'] == 'runtime smoke only; not benchmark evidence'
