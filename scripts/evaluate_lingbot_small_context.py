"""Released Gemini judgment of the frozen, scripted small-model context pair."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worldsync.gemini_vertex_judge import create_vertex_client, judge_episode


def main():
    if not json.loads((ROOT / 'reproduction/state.json').read_text())['gemini_calls_authorized']:
        raise ValueError('Gemini judging not authorized')
    output = ROOT / 'reproduction/lingbot-small'
    task = next(t for t in json.loads((ROOT / 'external/PlayWorld/data/gc/data.json').read_text())
                if t['task_id'] == 'GC027')
    plan = {'scope': 'Scripted single-case cache diagnostic, not adaptive PlayWorld or overall',
            'task_id': 'GC027', 'seed': 42, 'arms': {},
            'judge_label': 'world model', 'task_sha256': hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest()}
    for arm, capacity in [('cache18',18), ('cache70',70)]:
        run = json.loads((output / arm / 'run.json').read_text())
        if (run['status'] != 'completed' or run['task_id'] != 'GC027' or run['seed'] != 42
                or run['fixed_actions'] != ['WAIT'] + ['RIGHT']*12
                or run['local_attn_size'] != capacity or not run['native_fixed_controls']
                or run['generated_seconds'] != 12.8125):
            raise ValueError('Run differs from the declared pair')
        plan['arms'][arm] = {'video_sha256': hashlib.sha256((output/arm/'rollout.mp4').read_bytes()).hexdigest()}
    (output / 'judge-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    client = create_vertex_client()
    def evaluate(arm):
        result = judge_episode(client, ROOT/'external/PlayWorld', task, output/arm/'rollout.mp4',
                               ROOT/'external/PlayWorld/data'/task['image_path'],
                               world_model='world model', cache_dir=output/'judge-cache')
        (output/arm/'gemini-score.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        return arm, {'score':result['scores']['official'], 'gate':result['gate'],
                     'judge_model':result['judge_model'], 'video_sha256':result['video_sha256']}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(pool.map(evaluate, plan['arms']))
    report = {'scope':plan['scope'], 'complete':True, 'results':results,
              'caveat':'One seed and scripted camera path; inspect actual revisit and dynamic scene behavior. No overall or adaptive improvement claim.'}
    (output/'gemini-summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
