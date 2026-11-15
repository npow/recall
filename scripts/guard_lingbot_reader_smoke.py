"""Supervise frozen reader feasibility probes, retaining16GiB physical RAM."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESERVE = 16 * 1024**3
ESTIMATE = 90_000_000_000


def available():
    return int(next(line for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:')).split()[1]) * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real', action='store_true', help='Use the frozen6-train/2-held-out real-data plan')
    parser.add_argument('--preflight-only', action='store_true', help='Validate inputs without starting a GPU child')
    args = parser.parse_args()
    prefix = 'thor-reader-real' if args.real else 'thor-reader-gradient'
    output = ROOT / 'outputs/lingbot-reproduction' / f'{prefix}-{time.time_ns()}'
    output.mkdir(parents=True)
    report = {'scope': 'One-update synthetic reader plumbing; no quality claim',
              'available_before_bytes': available(), 'app_estimate_bytes': ESTIMATE,
              'system_reserve_bytes': RESERVE, 'output': str(output), 'status': 'not_launched'}
    extra = ['--synthetic-tasks', 'GC001', '--gradient-smoke']
    if args.real:
        from worldsync.lingbot_memory_data import read_manifest, split_manifest
        plan_path = ROOT / 'reproduction/learned-memory-reader/initial-real-data-plan.json'
        plan = json.loads(plan_path.read_text())
        data_root = ROOT / 'outputs/reader-train-data'
        manifest = data_root / 'selected.jsonl'
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        if digest(manifest) != plan['selected_manifest_sha256']:
            raise ValueError('selected manifest changed after freezing')
        for entry in plan['staged_files']:
            if digest(data_root / entry['path']) != entry['sha256']:
                raise ValueError('real training file changed')
        train, held = split_manifest(read_manifest(manifest, True), 2, 6)
        if ([e['clip_id'] for e in train] != plan['train_clip_ids'] or
                [e['clip_id'] for e in held] != plan['heldout_clip_ids']):
            raise ValueError('training/held-out split changed')
        config = plan['training_configuration']
        if config['adapter'] != 'memory-reader' or config['rank'] != 32:
            raise ValueError('unexpected reader architecture')
        extra = ['--manifest', str(manifest), '--data-root', str(data_root), '--content-ok-only',
                 '--heldout-count', '2', '--max-train', '6', '--steps', str(config['steps']),
                 '--lr', str(config['lr']), '--seed', str(config['seed']), '--quick-eval', '--release-text-encoder']
        report.update(scope='Frozen real-data reader feasibility, not a world-quality result',
                      plan_sha256=digest(plan_path), train_clips=len(train), heldout_clips=len(held))
    busy = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                          capture_output=True, text=True, check=True).stdout.strip()
    if args.preflight_only:
        report['status'] = 'preflight_passed'
    elif busy:
        report['status'] = 'gpu_busy_not_launched'
    elif report['available_before_bytes'] < ESTIMATE + RESERVE:
        report['status'] = 'budget_rejected'
    else:
        with (output / 'child.log').open('w') as log, (output / 'supervisor-samples.jsonl').open('w') as samples:
            command = [str(ROOT / '.venv-lingbot/bin/python'), '-u',
                       str(ROOT / 'scripts/train_lingbot_memory.py'), '--adapter', 'memory-reader',
                       '--rank', '32', *extra, '--parity',
                       '--output', str(output / 'training')]
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            report.update(status='running', pid=child.pid, command=command)
            (output / 'supervisor.json').write_text(json.dumps(report, indent=2) + '\n')
            minimum = report['available_before_bytes']
            while child.poll() is None:
                memory = available()
                minimum = min(minimum, memory)
                samples.write(json.dumps({'monotonic_s': time.monotonic(), 'mem_available_bytes': memory}) + '\n')
                samples.flush()
                if memory < RESERVE:
                    report.update(status='reserve_abort', reserve_abort_available_bytes=memory)
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    break
                time.sleep(.25)
            report.update(returncode=child.wait(), minimum_available_bytes=minimum)
            if report['status'] != 'reserve_abort':
                report['status'] = 'completed' if report['returncode'] == 0 else 'failed'
    (output / 'supervisor.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if report['status'] not in {'completed', 'preflight_passed'}:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
