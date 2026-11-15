"""Validate saved runtime-smoke evidence; never produces a benchmark score."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from worldsync.canonical_camera_plan import build_plan
from worldsync.camera_conditioning import plan_translation_scale
from worldsync.lingbot_controls import CameraPlan
from worldsync.native_reference_timeline import apply_camera_steps
from worldsync.lingbot_released_baseline import load_task, PROFILE


def validate(directory, source_root=None):
    directory = Path(directory)
    protocol = json.loads((directory/'protocol.json').read_text())
    episode = directory/'episode'
    report = json.loads((episode/'run.json').read_text())
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(protocol.get('scope') == 'runtime smoke only; not benchmark evidence', 'Wrong validation scope')
    require(protocol.get('generation_status') == report.get('status') == 'completed', 'Generation not completed')
    require(report.get('protocol_id') == PROFILE, 'Wrong controller protocol')
    require(report.get('task_id') == protocol['task_id'] == 'GC010' and report.get('split') == 'gc', 'Smoke requires GC010')
    require(report.get('text_prompt_mode') == 'prompt', 'Task prompt not used')
    require(report.get('local_attn_size') == 18 and report.get('sink_size') == 6, 'Non-stock cache')
    require(report.get('agent_interactions') == 0, 'GC010 smoke must not call hold service')
    require(report.get('generated_frames') == 29, 'Two native chunks must produce29frames')
    require(len(report.get('chunks', [])) == len(report.get('actions', [])) == 2, 'Expected exactly two complete chunks')
    require(report.get('controller_profile', {}).get('source_sha256') == protocol['source_sha256'], 'Inner and outer provenance disagree')
    root = Path(source_root) if source_root is not None else Path(__file__).resolve().parents[1]
    for name, expected in protocol['source_sha256'].items():
        require(hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, 'Source changed: '+name)
    _, timeline = build_plan(load_task('gc','GC010'))
    camera = CameraPlan()
    for index, (action, chunk) in enumerate(zip(report['actions'],report['chunks'])):
        expected = ([{'action':'WAIT','turn_degrees':0}]*3 if index == 0 else timeline.take())
        require(action.get('native_camera_steps') == expected, 'Camera schedule differs from canonical plan')
        require(action.get('translation_scale') == plan_translation_scale(timeline), 'Translation scale differs')
        apply_camera_steps(camera, expected)
        for field in ['position_normalized_units','accumulated_yaw_degrees','world_time_seconds']:
            require(np.allclose(action['pose_after'][field],camera.commanded_state()[field]), 'Recorded camera pose differs')
        movie = episode/chunk['path']
        require(hashlib.sha256(movie.read_bytes()).hexdigest() == chunk['sha256'], 'Chunk hash mismatch')
        observation = episode/f'observation-{index:03d}.png'
        with Image.open(observation) as image:
            require(image.format == 'PNG' and min(image.size)>0, 'Missing lossless observation')
            image.verify()
    probe = subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=nb_read_frames,r_frame_rate,width,height','-of','json',str(episode/'rollout.mp4')],text=True)
    stream = json.loads(probe)['streams'][0]
    require(int(stream['nb_read_frames']) == 29 and stream['r_frame_rate'] == '16/1', 'Saved video timing differs')
    return {'status':'validated','scope':'GC010 two-chunk runtime only; no visual return or benchmark claim',
            'frames':29,'source_sha256':protocol['source_sha256'],
            'video_sha256':hashlib.sha256((episode/'rollout.mp4').read_bytes()).hexdigest(),
            'final_commanded_state':camera.commanded_state(),'video_stream':stream}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,help='Archived source tree actually used by the run; all recorded hashes are verified')
    args=parser.parse_args()
    result=validate(args.directory,args.source_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'scope':result['scope']}))


if __name__=='__main__':
    main()
