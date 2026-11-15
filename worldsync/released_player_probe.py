"""Check one released hold-service request before paying for GPU generation."""
import argparse
import json
from pathlib import Path

from worldsync.credentials import load_local_credentials
from worldsync.lingbot_released_baseline import load_task, ROOT
from worldsync.canonical_camera_plan import build_plan
from worldsync.released_reference_player import ReleasedReferenceController


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path to preserve earlier evidence')
    task=load_task('if','IF001')
    actions,_=build_plan(task)
    load_local_credentials()
    from anthropic import Anthropic
    client=Anthropic(timeout=40,max_retries=0,default_headers={'Accept-Encoding':'identity'})
    result={'scope':'One live actor transport request on initial IF001 image; no GPU or benchmark score',
            'task_id':'IF001','status':'failed'}
    try:
        decision,metadata=ReleasedReferenceController().check_hold(
            client,task,actions,0,0,0,ROOT/'external/PlayWorld/data'/task['image_path'])
        result.update(status='passed',decision=decision,metadata=metadata)
    except Exception as error:
        # Keep provider exception text out of receipts: it may contain endpoint details.
        result.update(error_type=type(error).__name__)
        raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        client.close()
    print(json.dumps({'status':result['status'],'scope':result['scope']}))


if __name__=='__main__':
    main()
