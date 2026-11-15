"""Judge completed paired episodes with exactly three declared independent repeats."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from worldsync.gemini_vertex_judge import run_manifest
from worldsync.credentials import load_local_credentials
from scripts.run_codex_ccv import CASES

BASE=ROOT/'outputs/codex-ccv'
RUN=BASE/'artifacts/codex-ccv'
REPEATS=('rep1','rep2','rep3')


def ready(arm,split,task):
    status=RUN/f'{arm}-{split}-{task}-status.json'
    episode=RUN/arm/f'{split}-{task}'
    if arm=='C':episode=episode/'episode'
    try:
        state=json.loads(status.read_text())
        report=json.loads((episode/'run.json').read_text())
        if state['returncode']!=0 or report['status']!='completed':return None
        video=episode/'rollout.mp4'
        if not video.is_file():return None
        probe=subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=nb_read_frames','-of','json',str(video)],text=True,stderr=subprocess.DEVNULL)
        frames=int(json.loads(probe)['streams'][0]['nb_read_frames'])
        if frames != report['generated_frames']:return None
        return video
    except (OSError,ValueError,KeyError,subprocess.CalledProcessError):return None


def judge(identifier,split,task,video,rep):
    manifest=BASE/'manifests'/f'{identifier}-{rep}.jsonl'
    manifest.parent.mkdir(parents=True,exist_ok=True)
    dataset=json.loads((ROOT/'external/PlayWorld/data'/split/'data.json').read_text())
    record=next(t for t in dataset if t['task_id']==task)
    row={'episode_id':identifier,'split':split,'task_id':task,'video':str(video),
         'reference_image':str(ROOT/'external/PlayWorld/data'/record['image_path']),
         'world_model':'LingBot-World2 14B causal-fast'}
    manifest.write_text(json.dumps(row)+'\n')
    run_manifest(manifest,RUN/'judge'/rep,ROOT/'external/PlayWorld',workers=1)


def generation_accounted_for(scheduled):
    for split,tasks in CASES.items():
        for task in tasks:
            for arm in ['L','C']:
                identifier=f'{arm}-{split}-{task}'
                try:
                    status=json.loads((RUN/f'{identifier}-status.json').read_text())
                except (OSError,ValueError):return False
                if status['returncode']!=0:continue
                if not all((identifier,rep) in scheduled or
                           (RUN/'judge'/rep/'episodes'/f'{identifier}.json').exists() or
                           (BASE/'judge-errors'/f'{identifier}-{rep}.json').exists() for rep in REPEATS):
                    return False
    return True


def main():
    load_local_credentials()
    lease=json.loads((ROOT/'outputs/lambda/codex-ccv-lease.json').read_text())
    deadline=lease['terminate_after_unix']+3600
    scheduled=set();pending={}
    workers=int(os.environ.get('CCV_JUDGE_WORKERS','3'))
    if not 1 <= workers <= 12:raise ValueError('Judge concurrency must be between1and12')
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        while time.time()<deadline:
            for split,tasks in CASES.items():
                for task in tasks:
                    for arm in ['L','C']:
                        identifier=f'{arm}-{split}-{task}'
                        outstanding=[r for r in REPEATS if (identifier,r) not in scheduled and not (RUN/'judge'/r/'episodes'/f'{identifier}.json').exists() and not (BASE/'judge-errors'/f'{identifier}-{r}.json').exists()]
                        if not outstanding:continue
                        video=ready(arm,split,task)
                        if video is None:continue
                        for rep in outstanding:
                            scheduled.add((identifier,rep))
                            future=pool.submit(judge,identifier,split,task,video,rep)
                            pending[future]=(identifier,rep)
            for future in list(pending):
                if future.done():
                    identifier,rep=pending.pop(future)
                    try:future.result()
                    except Exception as error:
                        p=BASE/'judge-errors';p.mkdir(exist_ok=True)
                        (p/f'{identifier}-{rep}.json').write_text(json.dumps({'error_type':type(error).__name__})+'\n')
                        print(json.dumps({'event':'judge_worker_failed','episode':identifier,'rep':rep,'error_type':type(error).__name__}),flush=True)
            if (RUN/'selection.json').exists():
                subprocess.run([sys.executable,str(ROOT/'scripts/ccv_report.py'),'--run',str(RUN),
                                '--selection',str(RUN/'selection.json'),'--output',str(BASE/'comparison.md')],stdout=subprocess.DEVNULL)
            reports=list((RUN/'judge').glob('rep*/episodes/*.json'))
            (BASE/'judge-progress.json').write_text(json.dumps({'report_count':len(reports),'expected':72,'pending':len(pending),'updated_at':time.time()})+'\n')
            if len(reports)==72 and not pending:break
            if bool(list(RUN.glob('generation-terminal*.json'))) and not pending and generation_accounted_for(scheduled):
                # Every successful generation has been scheduled; failures stay missing.
                break
            time.sleep(30)


if __name__=='__main__':main()
