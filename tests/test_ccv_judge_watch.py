import json
from scripts import watch_codex_ccv_judging as watch


def test_terminal_marker_cannot_skip_successful_video_still_transferring(tmp_path,monkeypatch):
    monkeypatch.setattr(watch,'RUN',tmp_path)
    monkeypatch.setattr(watch,'CASES',{'gc':['GC024']})
    for arm in ['L','C']:
        (tmp_path/f'{arm}-gc-GC024-status.json').write_text(json.dumps({'returncode':0}))
    assert not watch.generation_accounted_for(set())
    scheduled={(f'{arm}-gc-GC024',rep) for arm in ['L','C'] for rep in watch.REPEATS}
    assert watch.generation_accounted_for(scheduled)


def test_generation_failures_remain_missing_without_blocking_final_report(tmp_path,monkeypatch):
    monkeypatch.setattr(watch,'RUN',tmp_path)
    monkeypatch.setattr(watch,'CASES',{'gc':['GC024']})
    for arm in ['L','C']:
        (tmp_path/f'{arm}-gc-GC024-status.json').write_text(json.dumps({'returncode':1}))
    assert watch.generation_accounted_for(set())
    assert not list(tmp_path.glob('judge/*/episodes/*.json'))


def test_failed_judge_attempts_survive_worker_restart_without_fabricated_scores(tmp_path,monkeypatch):
    monkeypatch.setattr(watch,'RUN',tmp_path/'run')
    monkeypatch.setattr(watch,'BASE',tmp_path)
    monkeypatch.setattr(watch,'CASES',{'gc':['GC024']})
    watch.RUN.mkdir()
    errors=tmp_path/'judge-errors';errors.mkdir()
    for arm in ['L','C']:
        identifier=f'{arm}-gc-GC024'
        (watch.RUN/f'{identifier}-status.json').write_text(json.dumps({'returncode':0}))
        for rep in watch.REPEATS:
            (errors/f'{identifier}-{rep}.json').write_text(json.dumps({'error_type':'transport_failure'}))
    assert watch.generation_accounted_for(set())
    assert not list(watch.RUN.glob('judge/*/episodes/*.json'))
