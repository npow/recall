"""Run only the one-update reader plumbing probe, retaining16GiB physical RAM."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RESERVE = 16 * 1024**3
ESTIMATE = 90_000_000_000


def available():
    return int(next(line for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:')).split()[1]) * 1024


def main():
    output = ROOT / 'outputs/lingbot-reproduction' / f'thor-reader-gradient-{time.time_ns()}'
    output.mkdir(parents=True)
    report = {'scope': 'One-update synthetic reader plumbing; no quality claim',
              'available_before_bytes': available(), 'app_estimate_bytes': ESTIMATE,
              'system_reserve_bytes': RESERVE, 'output': str(output), 'status': 'not_launched'}
    busy = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                          capture_output=True, text=True, check=True).stdout.strip()
    if busy:
        report['status'] = 'gpu_busy_not_launched'
    elif report['available_before_bytes'] < ESTIMATE + RESERVE:
        report['status'] = 'budget_rejected'
    else:
        with (output / 'child.log').open('w') as log, (output / 'supervisor-samples.jsonl').open('w') as samples:
            command = [str(ROOT / '.venv-lingbot/bin/python'), '-u',
                       str(ROOT / 'scripts/train_lingbot_memory.py'), '--adapter', 'memory-reader',
                       '--rank', '32', '--synthetic-tasks', 'GC001', '--gradient-smoke', '--parity',
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
    if report['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
