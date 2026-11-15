"""Sequential paired controller comparison on one GPU; each attempt is preserved."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
CASES={'gc':['GC024','GC022','GC030'],'if':['IF012','IF006','IF022'],
       'insight':['OE066','OE009','OE020'],'outsight':['OE055','OE028','OE062']}


def main():
    output=ROOT/'outputs/codex-ccv';output.mkdir(parents=True,exist_ok=True)
    selection={'rule':"Three cases per split, sha256('42:'+task_id), retained from Odin's declared plan",'picked':CASES,
               'seed':42,'legacy_commit':'5b8a5b8409a97fb609e72964f90b83d871ad6675',
               'comparison':'Complete legacy configuration versus corrected configuration; not isolated controller ablation',
               'judge_replicates':['rep1','rep2','rep3'],'gpu_policy':'Both arms on the same rented H100SXM; no mixed-device scores'}
    (output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    env=dict(os.environ,PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',WORLDSYNC_HOLD_MAX_TOKENS='400')
    jobs=[(split,task,arm) for i in range(3) for split in CASES for task in [CASES[split][i]] for arm in ['C','L']]
    shard=int(os.environ.get('CCV_SHARD','0'));shards=int(os.environ.get('CCV_SHARDS','1'))
    if not 0 <= shard < shards:raise ValueError('Invalid paired shard')
    jobs=[job for index,job in enumerate(jobs) if (index//2)%shards==shard]
    case_ids=os.environ.get('CCV_CASE_IDS')
    if case_ids:
        requested=set(case_ids.split(','))
        known={f'{split}-{task}' for split,task,_ in jobs}
        if not requested <= known:raise ValueError('Unknown case assignment')
        jobs=[job for job in jobs if f'{job[0]}-{job[1]}' in requested]
    for split,task,arm in jobs:
        episode=output/arm/f'{split}-{task}'
        status=output/f'{arm}-{split}-{task}-status.json'
        if status.exists():continue  # Never silently replace any attempt, including failures.
        if arm=='C':
            cwd=ROOT
            args=['-m','worldsync.lingbot_released_baseline','--split',split,'--task-id',task,'--seed','42',
                  '--chunks','60','--generate','--validate-controller','--output',str(episode)]
        else:
            cwd=ROOT/'legacy'
            args=['-m','worldsync.lingbot_rollout','--split',split,'--task-id',task,'--seed','42',
                  '--max-chunks','60' if split=='insight' else '40','--text-prompt','caption' if split=='insight' else 'both',
                  '--player-retries','2','--offload-text-encoder','--output',str(episode)]
        started=time.time()
        with (output/f'{arm}-{split}-{task}.log').open('w') as log:
            result=subprocess.run([sys.executable,*args],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
        record={'arm':arm,'split':split,'task_id':task,'returncode':result.returncode,'elapsed_seconds':time.time()-started,'argv':args}
        status.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)
    (output/f'generation-terminal-shard{shard}.json').write_text(json.dumps({'jobs_attempted':len(jobs),'completed_at':time.time()})+'\n')


if __name__=='__main__':main()
