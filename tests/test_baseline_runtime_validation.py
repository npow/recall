import hashlib
import json
import subprocess

import pytest
from PIL import Image

from worldsync.baseline_runtime_validation import validate
from worldsync.lingbot_released_baseline import preflight, BaselinePlayer, load_task
from worldsync.lingbot_controls import CameraPlan
from worldsync.native_reference_timeline import apply_camera_steps


@pytest.fixture
def evidence(tmp_path):
    episode=tmp_path/'episode';episode.mkdir()
    protocol=preflight('gc','GC010',chunks=2)
    protocol.update(scope='runtime smoke only; not benchmark evidence',generation_status='completed')
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    player=BaselinePlayer(None);camera=CameraPlan();actions=[];chunks=[]
    for index, frames in enumerate([13,16]):
        decision, metadata=player.decide(None,load_task('gc','GC010'),None,None,actions)
        apply_camera_steps(camera,metadata['native_camera_steps'])
        actions.append({**decision,**metadata,'pose_after':camera.commanded_state()})
        movie=episode/f'chunk-{index:03d}.mp4'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=size=32x32:rate=16',
                        '-frames:v',str(frames),str(movie)],check=True)
        Image.new('RGB',(32,32)).save(episode/f'observation-{index:03d}.png')
        chunks.append({'path':movie.name,'sha256':hashlib.sha256(movie.read_bytes()).hexdigest()})
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=size=32x32:rate=16',
                    '-frames:v','29',str(episode/'rollout.mp4')],check=True)
    report={'status':'completed','protocol_id':protocol['profile'],'task_id':'GC010','split':'gc',
            'text_prompt_mode':'prompt','local_attn_size':18,'sink_size':6,'agent_interactions':0,
            'generated_frames':29,'chunks':chunks,'actions':actions,'controller_profile':protocol}
    (episode/'run.json').write_text(json.dumps(report))
    return tmp_path


def test_complete_transport_fixture_validates_without_claiming_visual_success(evidence):
    result=validate(evidence)
    assert result['status']=='validated'
    assert 'no visual return' in result['scope']


@pytest.mark.parametrize('mutation',['missing_observation','corrupt_chunk','wrong_action'])
def test_incomplete_or_mismatched_evidence_rejected(evidence,mutation):
    episode=evidence/'episode'
    if mutation=='missing_observation':
        (episode/'observation-001.png').unlink()
    elif mutation=='corrupt_chunk':
        (episode/'chunk-001.mp4').write_bytes(b'changed')
    else:
        path=episode/'run.json';r=json.loads(path.read_text())
        r['actions'][1]['native_camera_steps'][0]['turn_degrees']=99
        path.write_text(json.dumps(r))
    with pytest.raises((ValueError,FileNotFoundError)):
        validate(evidence)
