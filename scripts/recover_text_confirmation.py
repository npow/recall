"""Recover the exact frozen confirmation cohort from committed Modal outputs; never regenerate/select retries."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'reproduction/task-text-confirmation'
    destination=output/'results.json'
    if destination.exists():
        raise ValueError('Canonical results already exist; refusing to overwrite')
    jobs=[json.loads(line) for line in (output/'jobs.jsonl').read_text().splitlines()]
    listing=json.loads(subprocess.check_output(['modal','volume','ls','--profile','nissan-pow',
                        'worldsync-lingbot-results','/','--json'],text=True))
    names=[entry['filename'] for entry in listing if entry['type']=='dir']
    selected={job['id']:[name for name in names if name.startswith(job['id']+'-')] for job in jobs}
    if any(len(matches)!=1 for matches in selected.values()):
        raise ValueError('Require exactly one committed attempt for every declared job, including failures')
    def one(job):
        folder=ROOT/'outputs/recover-text-confirmation'/job['id']
        folder.mkdir(parents=True,exist_ok=True)
        target=folder/'run.json'
        remote=selected[job['id']][0]
        subprocess.run(['modal','volume','get','--profile','nissan-pow','worldsync-lingbot-results',
                        remote+'/run.json',str(target)],check=True,stdout=subprocess.DEVNULL)
        run=json.loads(target.read_text())
        if run['status'] not in ('completed','failed','empty_rollout'):
            raise ValueError('Committed run has no terminal status')
        return {'id':job['id'],'result_directory':remote,'argv':job['argv'],'run':run,
                'status':run['status'],'error':run.get('error'),'weights_unchanged':None,
                'job_seconds':None,'client_wall_s':None,
                'chunk_pixel_sha256':[c['pixel_sha256'] for c in run.get('chunks',[])],
                'warm_runtime':run.get('warm_runtime'),'recovered_from_committed_volume':True}
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(one,jobs))
    report={'mode':'warm','gpu':'H200','containers_requested':2,'total_wall_s':None,
            'results':rows,'recovery':{'app_id':'ap-SUnpiWHFl5tbdCKw81HY8B','local_driver_exit_code':143,
            'regenerated_episodes':0,'attempt_selection':'exactly one committed directory for each frozen job ID',
            'argv_source':'Frozen job declarations; runtime output path is in result_directory',
            'unavailable_fields':'Client timings and returned weight fingerprint are not reconstructed or claimed',
            'required_next_step':'Run canonical evaluate_task_text_pilot.py over all16 outcomes/settings'}}
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'declared':len(jobs),'recovered':len(rows),'completed':sum(r['status']=='completed' for r in rows)}))


if __name__=='__main__':main()
