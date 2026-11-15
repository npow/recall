"""Release an owned CCV GPU only after its worker exits and evidence is copied."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/codex-ccv'
SECONDARY = {'gc-GC030', 'if-IF022', 'insight-OE020', 'outsight-OE062'}
THIRD = {'gc-GC022', 'if-IF006', 'insight-OE009'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('worker', choices=['primary', 'secondary', 'third'])
    args = parser.parse_args()
    secondary = args.worker == 'secondary'
    third = args.worker == 'third'
    lease_path = ROOT / 'outputs/lambda' / ('codex-ccv-third-lease.json' if third else 'codex-ccv-extra-lease.json' if secondary else 'codex-ccv-lease.json')
    lease = json.loads(lease_path.read_text())
    expected = 'e84fbb24a8a943aaa731fdf855bc09ec' if third else 'daa2be26cc0c4638ac2c9ec881b45160' if secondary else '7287b46fd8c14c2b80abc7a681412452'
    if lease['instance_id'] != expected or not lease['name'].startswith('worldsync-baseline-'):
        raise ValueError('This cleanup is restricted to the three recorded comparison leases')
    process = json.loads((BASE / ('third-remote-process.json' if third else 'extra-remote-process.json' if secondary else 'remote-process.json')).read_text())
    from run_codex_ccv import CASES
    all_cases = {f'{split}-{task}' for split, tasks in CASES.items() for task in tasks}
    cases = sorted(THIRD if third else SECONDARY if secondary else all_cases - SECONDARY - THIRD)
    ssh = ['ssh', '-i', str(Path.home()/'.ssh/id_ed25519'), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
           '-o', 'UserKnownHostsFile='+str(ROOT/'outputs/lambda/known_hosts'), 'ubuntu@'+lease['ip']]
    remote_root = '/home/ubuntu/worldsync/outputs/codex-ccv/'
    inspection = """import hashlib,json,subprocess
from pathlib import Path
root=Path('/home/ubuntu/worldsync/outputs/codex-ccv')
state=subprocess.run(['ps','-p',str(PID),'-o','stat='],capture_output=True,text=True)
if state.returncode not in (0,1):raise RuntimeError('Cannot inspect worker')
if state.stdout.strip() and not state.stdout.strip().startswith('Z'):
 print(json.dumps({'live':True}));raise SystemExit
hashes={}
for case in CASE_LIST:
 for arm in ['C','L']:
  status=root/f'{arm}-{case}-status.json';record=json.loads(status.read_text())
  if type(record.get('returncode')) is not int:raise RuntimeError('Attempt unaccounted for')
  paths=[status,root/f'{arm}-{case}.log']
  if record['returncode']==0:
   episode=root/arm/case
   if arm=='C':episode=episode/'episode'
   run=episode/'run.json'
   if json.loads(run.read_text())['status']!='completed':raise RuntimeError('Invalid completed attempt')
   paths.extend([run,episode/'rollout.mp4'])
  for p in paths:hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps({'live':False,'hashes':hashes}))
""".replace('PID', str(int(process['pid']))).replace('CASE_LIST', repr(cases))
    while time.time() < lease['terminate_after_unix']-90:
        result = subprocess.run(ssh+['python3 -'], input=inspection, text=True, capture_output=True)
        if result.returncode == 0:
            evidence = json.loads(result.stdout)
            if not evidence['live']:
                # Include only this worker's cases; never copy the other worker's reservation markers.
                patterns = [f'{arm}-{case}*' for case in cases for arm in ['C','L']]
                patterns += [f'{arm}/{case}/***' for case in cases for arm in ['C','L']]
                options = ['--include=*/'] + ['--include='+p for p in patterns] + ['--exclude=*']
                copied = subprocess.run(['rsync','-az',*options,'-e',shlex.join(ssh[:-1]),ssh[-1]+':'+remote_root,
                                         str(BASE/'artifacts/codex-ccv')+'/'])
                local = BASE/'artifacts/codex-ccv'
                verified = copied.returncode == 0 and all((local/p).is_file() and
                    hashlib.sha256((local/p).read_bytes()).hexdigest()==h for p,h in evidence['hashes'].items())
                if verified:
                    receipt = dict(evidence, instance_id=expected, cases=cases, verified_at_unix=time.time())
                    (BASE/f'{args.worker}-collection-verified.json').write_text(json.dumps(receipt,indent=2)+'\n')
                    subprocess.run([sys.executable,str(ROOT/'scripts/lambda_baseline_lease.py'),'terminate',
                                    '--lease',str(lease_path)],check=True)
                    print('Evidence verified and owned lease termination requested',expected,flush=True)
                    return
        time.sleep(30)
    print('Early cleanup window ended; original hard-lease watchdog remains responsible',flush=True)


if __name__ == '__main__':
    main()
