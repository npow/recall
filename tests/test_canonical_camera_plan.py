import numpy as np
import pytest

from worldsync.canonical_camera_plan import build_plan
from worldsync.lingbot_released_baseline import load_task
from worldsync.lingbot_controls import CameraPlan
from worldsync.native_reference_timeline import apply_camera_steps


@pytest.mark.parametrize('task_id,angles', [
    ('OE044',[180,180,90,90]), ('OE047',[90,90,40]), ('OE052',[90,90,90,90])])
def test_multi_return_turns_match_released_objective(task_id, angles):
    _, timeline = build_plan(load_task('outsight', task_id))
    assert [p['angle'] for p in timeline.phases if p['angle'] is not None] == angles
    camera=CameraPlan()
    apply_camera_steps(camera,[{'action':'WAIT','turn_degrees':0}]*3)
    for _ in range(30):
        apply_camera_steps(camera,timeline.take())
    expected=np.eye(4)
    if task_id=='OE047':
        c,s=np.cos(np.deg2rad(40)),np.sin(np.deg2rad(40))
        expected[:3,:3]=[[1,0,0],[0,c,-s],[0,s,c]]
    assert np.allclose(camera.poses[-1],expected)


def test_task_change_invalidates_manual_binding():
    task=load_task('outsight','OE047')
    task['prompt']=task['prompt'].replace('40','50')
    with pytest.raises(ValueError,match='no longer matches'):
        build_plan(task)
