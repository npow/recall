import json
from pathlib import Path
import subprocess
import sys

import pytest
from scripts.ccv_report import load


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
