import json
from pathlib import Path
import subprocess
import sys

import pytest
from scripts.ccv_report import load, completion_bounds, SPLITS


def test_missing_score_bounds_preserve_fixed_results_and_bound_unknown():
    picked = {s: [s+'1', s+'2', s+'3'] for s in SPLITS}
    judged = {f'{a}-{s}-{t}': [{'score': 2} for _ in range(3)]
              for a in ['L', 'C'] for s in SPLITS for t in picked[s]}
    assert completion_bounds(judged, picked)['difference'] == (0, 0)
    judged['L-gc-gc1'][1]['score'] = None
    bounds = completion_bounds(judged, picked)
    assert bounds['L'] == pytest.approx((2-1/36, 2+3/36))
    assert bounds['C'] == (2, 2)
    assert bounds['difference'] == pytest.approx((-3/36, 1/36))
    assert judged['L-gc-gc1'][1]['score'] is None


def test_no_observations_give_full_legal_range():
    picked = {s: [s+'1', s+'2', s+'3'] for s in SPLITS}
    bounds = completion_bounds({}, picked)
    assert bounds == {'L': (1, 5), 'C': (1, 5), 'difference': (-4, 4)}


def write(root,rep,score,**extra):
    p=root/'judge'/rep/'episodes';p.mkdir(parents=True,exist_ok=True)
    (p/'C-gc-GC024.json').write_text(json.dumps({'episode_id':'C-gc-GC024','scores':{'official':score},'video_sha256':'same',**extra}))


def test_exactly_three_replicates_and_no_error_scores(tmp_path):
    write(tmp_path,'rep1',2)
    write(tmp_path,'rep2',5,error='failure')
    write(tmp_path,'rep3',True)
    write(tmp_path,'rep4',5)
    rows=load(tmp_path)['C-gc-GC024']
    assert [r['score'] for r in rows]==[2,None,None]


def test_different_videos_cannot_be_averaged(tmp_path):
    write(tmp_path,'rep1',2)
    write(tmp_path,'rep2',3,video_sha256='different')
    with pytest.raises(ValueError,match='inconsistent'):
        load(tmp_path)


def test_partial_run_withholds_complete_overall(tmp_path):
    selection={'rule':'test','picked':{'gc':['GC024','GC022','GC030'],'if':['IF012','IF006','IF022'],
                                   'insight':['OE066','OE009','OE020'],'outsight':['OE055','OE028','OE062']}}
    path=tmp_path/'selection.json';path.write_text(json.dumps(selection))
    write(tmp_path,'rep1',5)
    result=subprocess.check_output([sys.executable,'scripts/ccv_report.py','--run',str(tmp_path),
        '--selection',str(path),'--output',str(tmp_path/'report.md')],text=True)
    result=json.loads(result)
    assert result['overall']=={'L':None,'C':None}
    assert result['ci']==[None,None]
    assert 'C=0/3' in (tmp_path/'report.md').read_text()


def test_partial_dimension_difference_uses_only_matching_cases(tmp_path):
    selection={'rule':'test','picked':{'gc':['GC024','GC022','GC030'],'if':['IF012','IF006','IF022'],
                                   'insight':['OE066','OE009','OE020'],'outsight':['OE055','OE028','OE062']}}
    path=tmp_path/'selection.json';path.write_text(json.dumps(selection))
    for rep in ['rep1','rep2','rep3']:
        folder=tmp_path/'judge'/rep/'episodes';folder.mkdir(parents=True)
        for episode,score in [('L-gc-GC024',3),('C-gc-GC024',5),('C-gc-GC030',1)]:
            (folder/(episode+'.json')).write_text(json.dumps({'episode_id':episode,'scores':{'official':score},'video_sha256':episode}))
    report=tmp_path/'report.md'
    subprocess.run([sys.executable,'scripts/ccv_report.py','--run',str(tmp_path),
                    '--selection',str(path),'--output',str(report)],check=True,capture_output=True)
    row=next(line for line in report.read_text().splitlines() if 'pairs=1/3' in line)
    assert row.endswith('| 3.00 | 3.00 | 2.00 |')
