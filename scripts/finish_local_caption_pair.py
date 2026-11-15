"""Wait for the declared local pair, verify originals, then judge three fixed repeats."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/local-caption-pair-OE020'


def main():
    # The existing generation archive also freezes the judge adapter for this run.
    sys.path.insert(0,str(BASE/'source'))
    from worldsync.gemini_vertex_judge import run_manifest
    from worldsync.credentials import load_local_credentials
    load_local_credentials()
    while True:
        state=json.loads((BASE/'manifest.json').read_text())
        if state['status']=='failed':
            raise RuntimeError('Generation failed; pair will not be scored')
        if state['status']=='completed':break
        # A stale manifest is not evidence of a live worker. Do not restart it.
        pid=state['jobs'][-1].get('pid') if state['jobs'] else None
        if pid and not Path(f'/proc/{pid}').exists():
            time.sleep(2)
            state=json.loads((BASE/'manifest.json').read_text())
            if state['status']=='running' and state['jobs'][-1].get('pid')==pid:
                raise RuntimeError('Generation process absent; inspect terminal state before proceeding')
        time.sleep(30)
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_local_caption_pair.py'),
                    '--output',str(BASE/'pair-verification.json')],check=True)
    receipt=json.loads((BASE/'pair-verification.json').read_text())
    # Require each predeclared manifest to identify exactly the two verified originals.
    for rep in ('rep1','rep2','rep3'):
        path=BASE/'judge-manifests'/f'{rep}.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if len(rows)!=2 or {r['episode_id'] for r in rows}!={'task-only-insight-OE020','caption-plus-task-insight-OE020'}:
            raise ValueError('Unexpected judge manifest')
        for row in rows:
            arm=row['episode_id'].removesuffix('-insight-OE020')
            if hashlib.sha256(Path(row['video']).read_bytes()).hexdigest()!=receipt['videos'][arm]['sha256']:
                raise ValueError('Manifest does not match verified original')
    def judge(rep):
        marker=BASE/'judge-manifests'/f'{rep}-started.json'
        try:
            with marker.open('x') as handle:json.dump({'started_at':time.time()},handle)
        except FileExistsError:
            return {'rep':rep,'status':'previous_attempt_retained'}
        try:
            run_manifest(BASE/'judge-manifests'/f'{rep}.jsonl',BASE/'judge'/rep,
                         ROOT/'external/PlayWorld',workers=2)
            return {'rep':rep,'status':'attempts_finished'}
        except Exception as error:
            result={'rep':rep,'status':'worker_failed','error_type':type(error).__name__}
            (BASE/'judge-manifests'/f'{rep}-error.json').write_text(json.dumps(result)+'\n')
            return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(judge,('rep1','rep2','rep3')))
    (BASE/'judging-terminal.json').write_text(json.dumps({'attempts':results,'finished_at':time.time()},indent=2)+'\n')


if __name__=='__main__':main()
