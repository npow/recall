import hashlib
import json
from pathlib import Path
import subprocess

import pytest
from scripts.verify_local_caption_pair import verify


@pytest.fixture
def pair(tmp_path):
    # Real tiny encoded videos exercise frame counting, rather than mocking ffprobe.
    (tmp_path/'source').mkdir();(tmp_path/'source/code.py').write_text('frozen source')
    source_hash=hashlib.sha256((tmp_path/'source/code.py').read_bytes()).hexdigest()
    arms=['task-only','caption-plus-task']
    (tmp_path/'manifest.json').write_text(json.dumps({'status':'completed','source_commit':'test',
        'jobs':[{'arm':a,'returncode':0} for a in arms]}))
    template=json.loads(Path('reproduction/baseline-gap/caption-runtime-smoke/run.json').read_text())
    for arm in arms:
        root=tmp_path/arm;episode=root/'episode';episode.mkdir(parents=True)
        video=episode/'rollout.mp4'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=size=16x16:rate=16',
                        '-frames:v','957','-c:v','libx264','-threads','1',str(video)],check=True)
        digest=hashlib.sha256(video.read_bytes()).hexdigest()
        run={**template,'actions':[template['actions'][0]]+[template['actions'][1]]*59,'generated_frames':957,'generated_seconds':59.8125,
             'text_prompt_mode':'prompt' if arm=='task-only' else 'both',
             'protocol_id':'lingbot-local-released-objective-reference-transfer-v1' if arm=='task-only' else 'lingbot-insight-caption-ablation-v1',
             'chunks':[{'path':'rollout.mp4','sha256':digest} for _ in range(60)]}
        (episode/'run.json').write_text(json.dumps(run))
        (root/'protocol.json').write_text(json.dumps({'source_sha256':{'code.py':source_hash}}))
    return tmp_path


def test_complete_pair_passes_and_hardware_mismatch_fails(pair):
    assert verify(pair)['status']=='passed_pair_integrity'
    path=pair/'caption-plus-task/episode/run.json';run=json.loads(path.read_text())
    run['hardware']={**run['hardware'],'gpu':'different GPU'};path.write_text(json.dumps(run))
    with pytest.raises(ValueError,match='hardware'):verify(pair)


def test_changed_source_is_rejected(pair):
    (pair/'source/code.py').write_text('changed')
    with pytest.raises(ValueError,match='Source changed'):verify(pair)


def test_pending_run_cannot_be_scored(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'status':'running'}))
    with pytest.raises(ValueError,match='not complete'):verify(tmp_path)
