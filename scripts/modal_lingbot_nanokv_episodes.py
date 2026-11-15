"""Frozen NanoKV episodes on reusable H200 workers; no default benchmark changes."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if modal.is_local():
    from scripts.modal_lingbot_benchmark import runtime_image, weights, results
    from worldsync.credentials import load_local_credentials
    load_local_credentials()
    import os
    image = (runtime_image.pip_install('anthropic==2.8.0').run_commands(
        'git init /work/external/NanoKV',
        'git -C /work/external/NanoKV remote add origin https://github.com/jiacheng-lu03/NanoKV.git',
        'git -C /work/external/NanoKV fetch --depth 1 origin a9baa538263e1c19acc05d10df80503c521c803c',
        'git -C /work/external/NanoKV checkout FETCH_HEAD')
        .env({'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'WORLDSYNC_LINGBOT_ATTENTION': 'sdpa'})
        .add_local_dir(ROOT / 'worldsync', '/work/worldsync', ignore=['__pycache__'])
        .add_local_file(ROOT / 'reproduction/lingbot-weights.json', '/work/reproduction/lingbot-weights.json')
        .add_local_file(ROOT / 'reproduction/state.json', '/work/reproduction/state.json')
        .add_local_dir(ROOT / 'external/PlayWorld/data', '/work/external/PlayWorld/data'))
else:
    image = None
    weights = modal.Volume.from_name('worldsync-lingbot-weights')
    results = modal.Volume.from_name('worldsync-lingbot-results')
secrets = [modal.Secret.from_dict({'ANTHROPIC_API_KEY': os.environ.get('ANTHROPIC_API_KEY', '')})]
app = modal.App('worldsync-lingbot-nanokv-episodes')


@app.function(image=image, cpu=2, memory=4096, timeout=180, volumes={'/cache': weights}, secrets=secrets)
def preflight(plan):
    import os
    from worldsync.lingbot_nanokv_episode import validate_plan
    validate_plan(plan)
    p = Path('/cache/nanokv/lingbot-world-2.0/codec.pt')
    if hashlib.sha256(p.read_bytes()).hexdigest() != 'b014909923f657b8019203bd581da77e8c6e0880bf7aa80b3eadfbf6a433f458':
        raise ValueError('Verified codec missing or changed')
    if any(not j.get('fixed_actions') for j in plan['jobs']) and not os.environ.get('ANTHROPIC_API_KEY'):
        raise ValueError('Adaptive player credentials absent')
    return {'status': 'passed', 'jobs': len(plan['jobs']), 'codec_sha256_verified': True}


@app.function(image=image, gpu='H200', cpu=8, memory=131072, timeout=7200,
              max_containers=6, volumes={'/cache': weights, '/results': results}, secrets=secrets)
def batch(plan, jobs):
    import contextlib
    import shutil
    import subprocess
    import traceback
    from worldsync import lingbot_rollout
    from worldsync.lingbot_nanokv_episode import run_episode, validate_plan
    from worldsync.lingbot_pipeline_reuse import PipelineReuse
    validate_plan(plan)
    if len({j['capacity_frames'] for j in jobs}) != 1:
        raise ValueError('A reusable worker requires one fixed cache capacity')
    weights.reload()
    link = Path('/work/external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast')
    link.parent.mkdir(exist_ok=True)
    if not link.exists():
        link.symlink_to('/cache/14b-causal-fast')
    identifier = 'nanokv-episodes-' + str(time.time_ns())
    root = Path('/work/outputs') / identifier
    root.mkdir(parents=True)
    (root / 'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    rows = []
    codec = Path('/cache/nanokv/lingbot-world-2.0/codec.pt')
    with (root / 'stdout.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        smoke = subprocess.run([sys.executable, '-u', '-m', 'worldsync.lingbot_nanokv_codec_smoke',
            '--codec', str(codec), '--source-root', '/work/external/NanoKV', '--device', 'cuda',
            '--output', str(root / 'codec-smoke.json')], stdout=log, stderr=subprocess.STDOUT, cwd='/work')
        if smoke.returncode:
            shutil.copytree(root, Path('/results') / identifier, dirs_exist_ok=True)
            results.commit()
            return {'status': 'failed', 'result_directory': identifier, 'error': 'GPU codec smoke failed', 'jobs': []}
        with PipelineReuse(lingbot_rollout) as worker:
            for job in jobs:
                folder = root / job['id']
                row = {'id': job['id'], 'result_directory': identifier + '/' + job['id']}
                try:
                    summary = run_episode(job, folder, codec_path=codec, source_root=Path('/work/external/NanoKV'))
                    row.update(status=summary['status'], worker_audit=worker.audit(), constructor_calls=worker.constructor_calls)
                except BaseException:
                    row.update(status='failed', error=traceback.format_exc())
                    worker.discard()
                rows.append(row)
                (root / 'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
                shutil.copytree(root, Path('/results') / identifier, dirs_exist_ok=True)
                results.commit()
                log.flush()
    return {'status': 'completed' if all(r['status']=='completed' for r in rows) else 'failed',
            'result_directory': identifier, 'jobs': rows}


@app.local_entrypoint()
def main(plan: str, output: str, containers: int = 2):
    from worldsync.lingbot_nanokv_episode import validate_plan
    from collections import defaultdict
    specification = json.loads((ROOT / plan).read_text())
    validate_plan(specification)
    directory = ROOT / output
    directory.mkdir(parents=True, exist_ok=False)
    checked = preflight.remote(specification)
    (directory / 'preflight.json').write_text(json.dumps(checked,indent=2)+'\n')
    grouped = defaultdict(list)
    for job in specification['jobs']:
        grouped[job['capacity_frames']].append(job)
    work = []
    for jobs in grouped.values():
        for i in range(min(max(1,containers),len(jobs))):
            part = jobs[i::min(max(1,containers),len(jobs))]
            work.append((specification, part))
    outcomes = list(batch.starmap(work))
    bundle = {'batches': outcomes, 'expected': len(specification['jobs']),
              'complete': all(o['status']=='completed' for o in outcomes)}
    (directory / 'results.json').write_text(json.dumps(bundle,indent=2)+'\n')
    for outcome in outcomes:
        for row in outcome['jobs']:
            local = directory / row['id']
            local.mkdir(exist_ok=True)
            for remote, name in [('summary.json','summary.json'), ('episode/run.json','run.json'), ('episode/rollout.mp4','rollout.mp4')]:
                try:
                    with (local / name).open('wb') as stream:
                        for part in results.read_file(row['result_directory'] + '/' + remote):
                            stream.write(part)
                except Exception as error:
                    (local / (name+'.download-error.txt')).write_text(str(error))
    if not bundle['complete']:
        raise RuntimeError('At least one episode failed; all remote artifacts retained')
