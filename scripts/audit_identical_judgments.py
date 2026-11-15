"""Audit repeated saved judgments sharing video/task/model/prompt identities; no API calls."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def audit(directory):
    groups=defaultdict(list)
    for path in sorted(Path(directory).glob('*.json')):
        result=json.loads(path.read_text())
        score=(result.get('scores') or {}).get('official')
        if result.get('error') or score is None:
            continue
        fields=['task_id','video_sha256','judge_model','judge_provider','gate_prompt_sha256','score_prompt_sha256']
        if not all(result.get(k) for k in fields[:4]):
            continue
        key=tuple(result.get(k) for k in fields)
        groups[key].append({'episode_id':result['episode_id'],'score':score,
                            'report_path':str(path),'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                            'served_models':sorted({c['served_model_version'] for c in result.get('calls',[]) if c.get('served_model_version')}),
                            'gate_verdict':(result.get('gate') or {}).get('verdict')})
    repeated=[]
    for key,rows in groups.items():
        if len(rows)<2:continue
        scores=[r['score'] for r in rows]
        repeated.append({**dict(zip(fields,key)),'reports':rows,'score_range':max(scores)-min(scores)})
    changed=[r for r in repeated if r['score_range']>0]
    return {'scope':'Saved-media and prompt-hash agreement; no newly generated or judged evidence',
            'limitations':['Historical reports do not record every request parameter or served version; this is not proof of byte-identical API requests.',
                           'Repeated groups are a post-hoc subset, not an unbiased estimate of whole-benchmark variance.',
                           'Existing scores remain unchanged; no favorable replicate selection.'],
            'repeated_groups':len(repeated),'groups_with_different_scores':len(changed),
            'maximum_observed_range':max((r['score_range'] for r in repeated),default=0),
            'groups':repeated}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes',type=Path,default=Path('reproduction/full1-recovered-judging/episodes'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.episodes)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='groups'}))
