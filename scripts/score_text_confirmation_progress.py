"""Score already committed confirmation episodes while generation continues; never report a partial mean."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from worldsync.gemini_vertex_judge import create_vertex_client, judge_episode


def main():
    output = ROOT/'reproduction/task-text-confirmation'
    plan = json.loads((output/'plan.json').read_text())
    jobs = [json.loads(line) for line in (output/'jobs.jsonl').read_text().splitlines()]
    if not json.loads((ROOT/'reproduction/state.json').read_text())['gemini_calls_authorized']:
        raise ValueError('Gemini judging not authorized')
    listing = json.loads(subprocess.check_output(['modal','volume','ls','--profile','nissan-pow',
                         'worldsync-lingbot-results','/','--json'],text=True))
    names = [row['filename'] for row in listing if row['type']=='dir']
    tasks = {t['task_id']:t for t in json.loads((ROOT/'external/PlayWorld/data/if/data.json').read_text())}
    client = create_vertex_client()
    def one(job):
        matches = [name for name in names if name.startswith(job['id']+'-')]
        if not matches:
            return {'id':job['id'],'status':'pending'}
        if len(matches)!=1:
            raise ValueError('Ambiguous attempt history; do not select a successful retry')
        argv=job['argv']; task_id=argv[argv.index('--task-id')+1]
        seed=int(argv[argv.index('--seed')+1]);mode=argv[argv.index('--text-prompt')+1]
        case=next(c for c in plan['cases'] if c['task_id']==task_id and c['seed']==seed)
        if hashlib.sha256(json.dumps(tasks[task_id],sort_keys=True).encode()).hexdigest()!=case['task_sha256']:
            raise ValueError('Frozen task changed')
        directory=output/job['id'];directory.mkdir(exist_ok=True)
        for filename in ('run.json','rollout.mp4'):
            destination=directory/filename
            if not destination.exists():
                subprocess.run(['modal','volume','get','--profile','nissan-pow','worldsync-lingbot-results',
                                matches[0]+'/'+filename,str(destination)],check=True,stdout=subprocess.DEVNULL)
        run=json.loads((directory/'run.json').read_text())
        if (run['status']!='completed' or run['task_id']!=task_id or run['seed']!=seed
                or run['text_prompt_mode']!=mode or run['initial_image_sha256']!=case['image_sha256']
                or run['native_fixed_controls'] or run['fixed_actions'] or run['split']!='if'
                or run['local_attn_size']!=18 or run['sink_size']!=6 or run['max_chunks']!=40
                or not run['player_streaming'] or any(a.get('player_response_pending') for a in run['actions'])):
            raise ValueError('Committed episode differs from frozen confirmation settings')
        score=judge_episode(client,ROOT/'external/PlayWorld',tasks[task_id],directory/'rollout.mp4',
                            ROOT/'external/PlayWorld/data'/tasks[task_id]['image_path'],
                            world_model='LingBot-World2 task conditioning trial',cache_dir=output/'cache')
        (directory/'gemini-score.json').write_text(json.dumps(score,indent=2,ensure_ascii=False)+'\n')
        return {'id':job['id'],'status':'scored','score':score['scores']['official'],'video_sha256':score['video_sha256']}
    def guarded(job):
        try:return one(job)
        except Exception as error:return {'id':job['id'],'status':'evaluation_failed','error':f'{type(error).__name__}: {error}'}
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(guarded,jobs))
    report={'scope':'Provisional per-episode judgments; canonical full manifest validation still required',
            'declared_episodes':len(jobs),'scored':sum(r['status']=='scored' for r in rows),'means':None,'results':rows}
    (output/'scoring-progress.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'scored':report['scored'],'declared':len(jobs),'means':None}))


if __name__=='__main__':main()
