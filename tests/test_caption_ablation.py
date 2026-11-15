import pytest

from worldsync import lingbot_rollout
from worldsync.lingbot_caption_ablation import prepare
from worldsync.lingbot_released_baseline import generation_argv, preflight


def test_ablation_changes_only_text_and_explicit_label(tmp_path):
    profile, argv = prepare('OE020', tmp_path)
    original = generation_argv('insight', 'OE020', tmp_path/'episode')
    expected = original.copy()
    expected[expected.index('--text-prompt')+1] = 'both'
    assert argv == expected+['--controlled-ablation']
    report = lingbot_rollout.main(argv+['--dry-run'],
        player_decide=lambda *a, **k: pytest.fail('No player in preflight'),
        simulation_chunk_budget=60, controller_profile=profile)
    assert report['text_prompt_mode'] == 'both'
    assert report['protocol_id'] == 'lingbot-insight-caption-ablation-v1'
    assert report['published_score_reproduced'] is False
    assert report['local_attn_size'] == 18 and report['sink_size'] == 6


def test_baseline_profile_still_rejects_caption_override(tmp_path):
    _, argv = prepare('OE020', tmp_path)
    with pytest.raises(ValueError, match='Canonical baseline requires'):
        lingbot_rollout.main(argv+['--dry-run'], player_decide=lambda *a: None,
            simulation_chunk_budget=60, controller_profile=preflight('insight', 'OE020'))


def test_ablation_requires_explicit_flag(tmp_path):
    profile, argv = prepare('OE020', tmp_path)
    argv.remove('--controlled-ablation')
    with pytest.raises(ValueError, match='explicit in-sight'):
        lingbot_rollout.main(argv+['--dry-run'], player_decide=lambda *a: None,
            simulation_chunk_budget=60, controller_profile=profile)
