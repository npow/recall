"""One frozen real-data reader fit on H200; CPU hash/split preflight first. No API calls."""
import hashlib
import json
from pathlib import Path
import sys

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if modal.is_local():
    from scripts.modal_lingbot_benchmark import runtime_image
    image = (runtime_image
        .add_local_dir(ROOT / 'worldsync', '/work/worldsync', ignore=['__pycache__'])
        .add_local_file(ROOT / 'scripts/train_lingbot_memory.py', '/work/scripts/train_lingbot_memory.py')
        .add_local_file(ROOT / 'outputs/reader-train-data/selected.jsonl', '/work/selected.jsonl')
        .add_local_file(ROOT / 'reproduction/learned-memory-reader/initial-real-data-plan.json', '/work/reader-plan.json')
        .add_local_file(ROOT / 'reproduction/lingbot-weights.json', '/work/reproduction/lingbot-weights.json')
        .add_local_file(ROOT / 'external/PlayWorld/data/gc/data.json', '/work/external/PlayWorld/data/gc/data.json'))
else:
    image = None
weights = modal.Volume.from_name('worldsync-lingbot-weights')
results = modal.Volume.from_name('worldsync-lingbot-results')
data = modal.Volume.from_name('worldsync-train-data')
app = modal.App('worldsync-lingbot-reader-fit')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_inputs():
    from worldsync.lingbot_memory_data import read_manifest, split_manifest
    data.reload()
    plan = json.loads(Path('/work/reader-plan.json').read_text())
    if digest('/work/selected.jsonl') != plan['selected_manifest_sha256']:
        raise ValueError('Frozen manifest changed')
    for row in plan['staged_files']:
        if digest(Path('/data') / row['path']) != row['sha256']:
            raise ValueError(f"Frozen input changed: {row['path']}")
    train, held = split_manifest(read_manifest('/work/selected.jsonl', True), 2, 6)
    if ([e['clip_id'] for e in train] != plan['train_clip_ids'] or
            [e['clip_id'] for e in held] != plan['heldout_clip_ids']):
        raise ValueError('Frozen source split changed')
    expected = {'adapter':'memory-reader','rank':32,'steps':60,'lr':.0001,'seed':0,
                'heldout_count':2,'max_train':6,'quick_eval':True,'release_text_encoder':True,'parity':True}
    if any(plan['training_configuration'].get(k) != value for k,value in expected.items()):
        raise ValueError('Job differs from frozen training settings')
    return {'plan_sha256':digest('/work/reader-plan.json'), 'files_verified':len(plan['staged_files']),
            'train_clip_ids':plan['train_clip_ids'], 'heldout_clip_ids':plan['heldout_clip_ids']}


@app.function(image=image, cpu=2, memory=4096, timeout=600, volumes={'/data':data})
def preflight():
    return validate_inputs()


@app.function(image=image, gpu='H200', cpu=8, memory=196608, timeout=7200, max_containers=1,
              volumes={'/data':data,'/cache':weights,'/results':results})
def fit():
    import os
    import shutil
    import subprocess
    import time
    verified = validate_inputs()
    weights.reload()
    target = Path('/work/external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast')
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to('/cache/14b-causal-fast')
    identifier = f'reader-real-{time.time_ns()}'
    output = Path('/results') / identifier
    output.mkdir(parents=True)
    command = [sys.executable,'-u','/work/scripts/train_lingbot_memory.py','--adapter','memory-reader',
               '--rank','32','--steps','60','--lr','0.0001','--seed','0','--quick-eval',
               '--release-text-encoder','--manifest','/work/selected.jsonl','--data-root','/data',
               '--content-ok-only','--heldout-count','2','--max-train','6','--parity',
               '--output',str(output/'training')]
    manifest = {'result_directory':identifier,'scope':'Frozen6/2 real-data feasibility, not PlayWorld',
                'preflight':verified,'command':command,'status':'running'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copyfile('/work/reader-plan.json',output/'plan.json')
    results.commit()
    try:
        with (output/'child.log').open('w') as log:
            child = subprocess.run(command,cwd='/work',stdout=log,stderr=subprocess.STDOUT,
                                   env={**os.environ,'PYTHONPATH':'/work','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'})
        manifest.update(returncode=child.returncode,status='completed' if child.returncode==0 else 'failed')
    except BaseException as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        results.commit()
    if (output/'training/train.json').exists():
        manifest['train'] = json.loads((output/'training/train.json').read_text())
    return manifest


@app.local_entrypoint()
def main():
    print(json.dumps({'preflight':preflight.remote()}),flush=True)
    outcome = fit.remote()
    destination = ROOT/'reproduction/learned-memory-reader/real-fit-result.json'
    destination.write_text(json.dumps(outcome,indent=2)+'\n')
    print(json.dumps({'status':outcome['status'],'result_directory':outcome['result_directory']}),flush=True)
