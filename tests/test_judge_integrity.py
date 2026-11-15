import json
from pathlib import Path

import pytest

from worldsync.judge_integrity import request_identity, validate_gate
from worldsync import gemini_vertex_judge as judge

PLAYWORLD = Path(__file__).resolve().parents[1] / 'external/PlayWorld'


def test_request_fingerprint_covers_evidence_task_label_and_transport(tmp_path):
    video = tmp_path / 'video'; video.write_bytes(b'video1')
    image = tmp_path / 'image'; image.write_bytes(b'image1')
    task = {'task_id': 'GC010', 'prompt': 'Turn'}
    def identity(**changes):
        args = dict(playworld=PLAYWORLD, task=task, video=video, reference_image=image,
                    world_model='LingBot', model='judge', transport='vertex')
        args.update(changes)
        return request_identity(**args)['fingerprint']
    original = identity()
    assert identity() == original
    for changes in [dict(task={**task, 'prompt': 'Wait'}), dict(world_model='Other'),
                    dict(model='other'), dict(transport='released'), dict(reference_image=None)]:
        assert identity(**changes) != original
    video.write_bytes(b'video2'); assert identity() != original
    video.write_bytes(b'video1'); image.write_bytes(b'image2'); assert identity() != original


@pytest.mark.parametrize('gate', [
    {'verdict': 'pass', 'instruction_following_score': 0},
    {'verdict': 'fail', 'instruction_following_score': 1},
    {'verdict': 'partial', 'instruction_following_score': True},
    {'verdict': 'pass', 'instruction_following_score': 1, 'task_id': 'GC011'},
    {'verdict': 'pass', 'instruction_following_score': float('nan')},
])
def test_inconsistent_gate_is_rejected(gate):
    with pytest.raises(ValueError):
        validate_gate(gate, 'GC010')


def test_missing_new_input_preserves_old_report_and_does_not_abort_manifest(tmp_path, monkeypatch):
    out = tmp_path / 'out'; (out / 'episodes').mkdir(parents=True)
    old = {'episode_id': 'a', 'task_id': 'GC010', 'scores': {'official': 4}}
    (out / 'episodes/a.json').write_text(json.dumps(old))
    monkeypatch.setattr(judge, 'create_vertex_client', lambda *a, **k: object())
    rows = [dict(episode_id=name, split='gc', task_id='GC010', video=str(tmp_path / 'missing')) for name in ['a', 'b']]
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('\n'.join(json.dumps(row) for row in rows))
    results = judge.run_manifest(manifest, out, PLAYWORLD, workers=1)
    assert len(results) == 2 and all('FileNotFoundError' in r['error'] for r in results)
    archives = list((out / 'previous-reports/a').glob('*.json'))
    assert len(archives) == 1 and json.loads(archives[0].read_text()) == old


def test_duplicate_episode_ids_fail_without_creating_client(tmp_path, monkeypatch):
    monkeypatch.setattr(judge, 'create_vertex_client', lambda *a, **k: pytest.fail('API client must not be created'))
    manifest = tmp_path / 'manifest.jsonl'
    row = dict(episode_id='same', split='gc', task_id='GC010', video='missing')
    manifest.write_text(json.dumps(row)+'\n'+json.dumps(row))
    with pytest.raises(ValueError, match='duplicate'):
        judge.run_manifest(manifest, tmp_path / 'out', PLAYWORLD)


@pytest.mark.parametrize('score', [True, float('nan'), float('inf'), 0, 6, '4'])
def test_bad_rubric_score_is_rejected_before_cache_reuse(score):
    judge.load_playworld(PLAYWORLD)
    with pytest.raises(ValueError):
        judge.validate_rubric({'task_id': 'GC010', 'final_score_1_to_5': score}, 'GC010')


def test_valid_rubric_keeps_released_weighted_aggregation():
    judge.load_playworld(PLAYWORLD)
    rubric = {'task_id': 'GC010', 'final_score_1_to_5': 3,
              'categories': {'identity_id': {'score': 4, 'weight': 1}, 'other': {'score': 1, 'weight': 1}}}
    judge.validate_rubric(rubric, 'GC010')
    assert judge.effective_scores(None, rubric)['official'] == 3
