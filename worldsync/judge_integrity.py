"""Immutable request identities for judge reuse; never treats an old score as new evidence."""
import hashlib
import json
from pathlib import Path


def sha_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):digest.update(data)
    return digest.hexdigest()


def request_identity(playworld, task, video, reference_image, world_model, model, transport='vertex'):
    root=Path(playworld)
    files=['metrics/vqa/rubric.py','metrics/vqa/instruction_gate.py','metrics/vqa/sampling.py','metrics/vqa/score.py','metrics/aggregate.py']
    result={'version':3,'video_sha256':sha_file(video),'reference_sha256':sha_file(reference_image) if reference_image else None,
            'task_sha256':hashlib.sha256(json.dumps(task,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'world_model':world_model,'judge_model':model,'transport':transport,
            'generation_config':{'response_mime_type':'application/json','temperature':.1},
            'released_source_sha256':{name:sha_file(root/name) for name in files},
            'adapter_source_sha256':{name:sha_file(Path(__file__).parent/name)
                                     for name in ['judge_integrity.py','gemini_vertex_judge.py']}}
    result['fingerprint']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return result


def validate_gate(gate, expected_task_id):
    if gate.get('task_id') not in [None,expected_task_id]:
        raise ValueError('Gate returned a different task ID')
    verdict=gate.get('verdict')
    if verdict not in ['pass','partial','fail']:
        raise ValueError('Applicable trajectory gate did not return pass/partial/fail')
    value=gate.get('instruction_following_score')
    if isinstance(value,bool):
        raise ValueError('Gate score must be numeric0or1, notBoolean')
    try:value=float(value)
    except (TypeError,ValueError):raise ValueError('Applicable gate omitted numeric score') from None
    if value != (0 if verdict=='fail' else 1):
        raise ValueError('Gate verdict contradicts numeric aggregation score')


def atomic_json(path, value):
    import os
    import tempfile
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor, temporary=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(descriptor,'w') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2)
            stream.write('\n')
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
