"""One frozen NanoKV proof arm on H200; CPU-staged codec, no player/judge calls."""
import hashlib
import json
from pathlib import Path
import sys
import time
import modal

ROOT = Path(__file__).resolve().parents[1]
NANOKV_REVISION = 'a9baa538263e1c19acc05d10df80503c521c803c'
CODEC_REVISION = '4b0dac1219ed72ead539f8d063348949e6aeb045'
CODEC_SHA = 'b014909923f657b8019203bd581da77e8c6e0880bf7aa80b3eadfbf6a433f458'
sys.path.insert(0, str(ROOT))
if modal.is_local():
    from scripts.modal_lingbot_benchmark import runtime_image, weights, results
    image = (runtime_image.run_commands(
        'git init /work/external/NanoKV',
        'git -C /work/external/NanoKV remote add origin https://github.com/jiacheng-lu03/NanoKV.git',
        f'git -C /work/external/NanoKV fetch --depth 1 origin {NANOKV_REVISION}',
        'git -C /work/external/NanoKV checkout FETCH_HEAD')
        .add_local_dir(ROOT / 'worldsync', '/work/worldsync', ignore=['__pycache__'])
        .add_local_file(ROOT / 'reproduction/lingbot-weights.json', '/work/reproduction/lingbot-weights.json')
        .add_local_file(ROOT / 'reproduction/full-history-GC010/run.json', '/work/reproduction/full-history-GC010/run.json')
        .add_local_file(ROOT / 'external/PlayWorld/data/gc/data.json', '/work/external/PlayWorld/data/gc/data.json')
        .add_local_dir(ROOT / 'external/PlayWorld/data/gc/images', '/work/external/PlayWorld/data/gc/images'))
else:
    image = None
    weights = modal.Volume.from_name('worldsync-lingbot-weights')
    results = modal.Volume.from_name('worldsync-lingbot-results')
app = modal.App('worldsync-lingbot-nanokv-proof')


@app.function(image=image, cpu=4, memory=8192, timeout=900, volumes={'/cache': weights})
def stage():
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download('jiacheng1234/NanoKV', 'lingbot-world-2.0/codec.pt',
        revision=CODEC_REVISION, local_dir='/cache/nanokv'))
    if hashlib.sha256(path.read_bytes()).hexdigest() != CODEC_SHA:
        raise ValueError('Released codec hash mismatch')
    weights.commit()
    return {'codec_path': str(path), 'sha256': CODEC_SHA, 'bytes': path.stat().st_size}


@app.function(image=image, gpu='H200', cpu=8, memory=131072, timeout=2400,
              max_containers=1, volumes={'/cache': weights, '/results': results})
def probe(plan, arm):
    import os
    import shutil
    import subprocess
    weights.reload()
    target = Path('/work/external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast')
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        target.symlink_to('/cache/14b-causal-fast')
    identifier = f'nanokv-{arm}-{time.time_ns()}'
    root = Path('/work/outputs') / identifier
    root.mkdir(parents=True)
    specification = root / 'plan.json'
    specification.write_text(json.dumps(plan, indent=2) + '\n')
    try:
        with (root / 'stdout.log').open('w') as log:
            outcome = subprocess.run([sys.executable, '-u', '-m', 'worldsync.lingbot_nanokv_probe',
                '--plan', str(specification), '--arm', arm, '--output', str(root / 'proof'),
                '--codec', '/cache/nanokv/lingbot-world-2.0/codec.pt'],
                cwd='/work', stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                         WORLDSYNC_LINGBOT_ATTENTION='sdpa'))
    finally:
        shutil.copytree(root, Path('/results') / identifier, dirs_exist_ok=True)
        results.commit()
    summary = root / 'proof/summary.json'
    run = root / 'proof/episode/run.json'
    return {'returncode': outcome.returncode, 'result_directory': identifier,
            'summary': json.loads(summary.read_text()) if summary.exists() else {'status': 'failed'},
            'run': json.loads(run.read_text()) if run.exists() else None}


@app.local_entrypoint()
def main(plan: str, arm: str, output: str):
    directory = ROOT / output
    directory.mkdir(parents=True, exist_ok=False)
    specification = json.loads((ROOT / plan).read_text())
    if arm not in ['identity58', 'codec18', 'codec58']:
        raise ValueError('Unknown arm')
    staged = stage.remote()
    (directory / 'staging.json').write_text(json.dumps(staged, indent=2) + '\n')
    result = probe.remote(specification, arm)
    run = result.pop('run')
    if run is not None:
        (directory / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
        if run.get('chunks'):
            with (directory / 'rollout.mp4').open('wb') as movie:
                for part in results.read_file(result['result_directory'] + '/proof/episode/rollout.mp4'):
                    movie.write(part)
    (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    if result['returncode'] or result['summary']['status'] != 'completed':
        raise RuntimeError('Proof arm failed; complete artifacts retained on results volume')
