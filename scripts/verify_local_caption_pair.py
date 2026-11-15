"""Verify the declared matched OE020 pair before judging; never modify originals."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def verify(base):
    manifest=json.loads((base/'manifest.json').read_text())
    if manifest['status']!='completed':
        raise ValueError('Pair is not complete; no scoring permitted')
    arms=('task-only','caption-plus-task')
    if [j['arm'] for j in manifest['jobs']]!=list(arms) or any(j['returncode']!=0 for j in manifest['jobs']):
        raise ValueError('Expected two successful generation attempts in declared order')
    runs={};profiles={};videos={}
    for arm in arms:
        episode=base/arm/'episode'
        r=runs[arm]=json.loads((episode/'run.json').read_text())
        p=profiles[arm]=json.loads((base/arm/'protocol.json').read_text())
        if (r['status']!='completed' or r['seed']!=42 or r['task_id']!='OE020'
                or r['generated_frames']!=957 or r['generated_seconds']!=59.8125
                or len(r['chunks'])!=60 or len(r['actions'])!=60 or r['agent_interactions']!=0):
            raise ValueError('Incomplete or altered generation protocol: '+arm)
        expected_text='prompt' if arm=='task-only' else 'both'
        expected_profile='lingbot-local-released-objective-reference-transfer-v1' if arm=='task-only' else 'lingbot-insight-caption-ablation-v1'
        if r['text_prompt_mode']!=expected_text or r['protocol_id']!=expected_profile:
            raise ValueError('Wrong arm identity: '+arm)
        for name,expected in p['source_sha256'].items():
            actual=hashlib.sha256((base/'source'/name).read_bytes()).hexdigest()
            if actual!=expected:raise ValueError('Source changed: '+name)
        video=episode/'rollout.mp4'
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=nb_read_frames,r_frame_rate,duration','-of','json',str(video)],text=True))['streams'][0]
        if int(probe['nb_read_frames'])!=957 or probe['r_frame_rate']!='16/1' or float(probe['duration'])!=59.8125:
            raise ValueError('Encoded output does not match full planned episode: '+arm)
        for chunk in r['chunks']:
            if hashlib.sha256((episode/chunk['path']).read_bytes()).hexdigest()!=chunk['sha256']:
                raise ValueError('Chunk changed: '+arm)
        videos[arm]={'path':str(video),'sha256':hashlib.sha256(video.read_bytes()).hexdigest(),'ffprobe':probe}
    left,right=(runs[a] for a in arms)
    for key in ['hardware','local_attn_size','sink_size','sampler_timesteps_index','sample_shift','quantization','decision_interval','seed','max_area','chunk_size']:
        if key not in left or key not in right or left[key]!=right[key]:
            raise ValueError('Unmatched setting: '+key)
    if left['local_attn_size']!=18 or left['sink_size']!=6 or left['quantization']!='none':
        raise ValueError('Expected stock18/6cache and unquantized weights')
    if [a['native_camera_steps'] for a in left['actions']]!=[a['native_camera_steps'] for a in right['actions']]:
        raise ValueError('Camera command schedules differ')
    common=set(profiles[arms[0]]['source_sha256']) & set(profiles[arms[1]]['source_sha256'])
    for name in common:
        if profiles[arms[0]]['source_sha256'][name]!=profiles[arms[1]]['source_sha256'][name]:
            raise ValueError('Source differs between arms: '+name)
    return {'status':'passed_pair_integrity','source_commit':manifest['source_commit'],'videos':videos,
            'hardware':left['hardware'],'scope':'protocol and artifact verification only; not quality or score improvement'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'outputs/local-caption-pair-OE020')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    try:result=verify(args.run)
    except (ValueError,KeyError,OSError) as error:parser.exit(2,str(error)+'\n')
    payload=json.dumps(result,indent=2)+'\n'
    if args.output:args.output.write_text(payload)
    print(payload)


if __name__=='__main__':main()
