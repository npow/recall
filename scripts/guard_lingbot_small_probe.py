"""Supervise only the isolated small-model child, preserving16GiB shared RAM."""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
RESERVE=16*1024**3
APP_ESTIMATE=45_000_000_000

def available():
    return int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-frames',type=int,choices=(18,70),required=True)
    a=p.parse_args()
    output=ROOT/'outputs/lingbot-reproduction'/f'thor-small-cache{a.cache_frames}-{time.time_ns()}'
    output.mkdir()
    report={'scope':'Independent official small-model baseline/context probe','cache_frames':a.cache_frames,
            'available_before_bytes':available(),'app_estimate_bytes':APP_ESTIMATE,
            'system_reserve_bytes':RESERVE,'output':str(output),'status':'not_launched'}
    busy=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    if busy:report['status']='gpu_busy_not_launched'
    elif report['available_before_bytes']<APP_ESTIMATE+RESERVE:report['status']='budget_rejected'
    else:
        with (output/'child.log').open('w') as log,(output/'supervisor-samples.jsonl').open('w') as samples:
            child=subprocess.Popen([str(ROOT/'.venv-lingbot/bin/python'),'-u','-m','worldsync.lingbot_small_probe',
                '--output',str(output),'--cache-frames',str(a.cache_frames)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            report.update(status='running',pid=child.pid)
            (output/'supervisor.json').write_text(json.dumps(report,indent=2)+'\n')
            while child.poll() is None:
                memory=available()
                samples.write(json.dumps({'monotonic_s':time.monotonic(),'mem_available_bytes':memory})+'\n');samples.flush()
                if memory<RESERVE:
                    report.update(status='reserve_abort',reserve_abort_available_bytes=memory)
                    child.terminate()
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    break
                time.sleep(.25)
            report['returncode']=child.wait()
            if report['status']!='reserve_abort':report['status']='completed' if report['returncode']==0 else 'failed'
    (output/'supervisor.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if report['status']!='completed':raise SystemExit(1)

if __name__=='__main__':main()
