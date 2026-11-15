"""Explicit closest-release stock baseline; legacy results remain separate."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'lingbot-local-released-objective-reference-transfer-v1'
GENERATION_BLOCKER = ('Canonical controller and fixed-scale conditioning are integrated; '
                      'full rollout hook execution and output provenance still require runtime validation.')


def load_task(split, task_id):
    path = ROOT / 'external/PlayWorld/data' / split / 'data.json'
    return next(t for t in json.loads(path.read_text()) if t['task_id']==task_id)


class BaselinePlayer:
    """Own canonical timing; use the released service only for visual hold checks."""
    def __init__(self, controller, simulation_chunks=60):
        self.controller, self.simulation_chunks = controller, simulation_chunks
        self.requests = 0
        self.timeline = None
        self.actions = None
        self.scale = None

    def decide(self, client, task, initial, latest, history, maximum=40, commanded_state=None):
        from worldsync.released_reference_player import canonical_actions
        from worldsync.native_reference_timeline import NativeReferenceTimeline
        from worldsync.camera_conditioning import plan_translation_scale
        if self.timeline is None:
            self.actions = canonical_actions(task)
            self.timeline = NativeReferenceTimeline(self.actions, task['prompt'])
            self.scale = plan_translation_scale(self.timeline)
        metadata = {'agent_interaction_increment': 0, 'translation_scale': self.scale}
        if not history:
            steps = [{'action': 'WAIT', 'turn_degrees': 0}] * 3
            metadata['bootstrap'] = True
        else:
            if self.timeline.check_due():
                if self.requests >= maximum:
                    return {'operation': 'END', 'action': None, 'reason': 'Haiku request budget exhausted'}, metadata
                phase = self.timeline.current()
                self.requests += 1
                decision, service = self.controller.check_hold(
                    client, task, self.actions, phase['source']['index'],
                    phase['used'] * 250, phase['extension'] * 250, latest)
                self.timeline.apply_decision(decision)
                metadata.update(service, agent_interaction_increment=1)
            steps = self.timeline.take(4)
        metadata.update(native_camera_steps=steps, haiku_requests_total=self.requests,
                        plan_complete=self.timeline.current() is None)
        return {'operation': 'KEEP', 'action': steps[0]['action'],
                'reason': 'Canonical quarter-second schedule; no visual success assertion'}, metadata


def generation_argv(split, task_id, output, *, seed=42, chunks=60):
    if type(chunks) is not int or not 1 <= chunks <= 60:
        raise ValueError('Simulation budget must be1..60chunks')
    return ['--split',split,'--task-id',task_id,'--seed',str(seed),'--max-chunks',str(chunks),
            '--text-prompt','prompt','--local-attn-size','18','--sink-size','6',
            '--decision-interval','1','--turn-degrees-per-second','60',
            '--denoise-schedule','stock','--turn-clamp','off','--offload-text-encoder',
            '--output',str(output)]


def preflight(split, task_id, *, seed=42, chunks=60):
    from worldsync.released_reference_player import ReleasedReferenceController
    from worldsync.reference_player import PlayerResponseError
    task = load_task(split,task_id)
    controller = ReleasedReferenceController()
    result = {'profile':PROFILE,'task_id':task_id,'split':split,'seed':seed,
        'world_text':task['prompt'],'world_text_source':'task.prompt exactly; no static-caption substitution',
        'simulation_chunks':chunks,'max_haiku_requests':40,
        'controller':controller.hold_service_metadata(),'stock_cache':{'frames':18,'sink':6},
        'backbone_modified':False,'training':False,'streamed_legacy_player':False,
        'exact_published_baseline_reproduced':False,
        'judge':{'requested_model':'gemini-3.1-pro','transport':'released Gemini API client',
                 'silent_alias_fallback':False},
        'native_capability':'Camera rays only; no trained avatar-action channel is invented.',
        'remaining_differences':['Hosted service intentionally excluded.',
            'Paper LingBot-specific controller/native keyboard timing is not released; HY reference behavior is transferred explicitly.',
            'Native frame-time chunk quantization and paused API calls differ from browser wall-clock control.',
            '40Haiku requests is an explicit local budget definition, not verified paper LingBot interaction counting.',
            'Exact judge alias currently404 with configured released client; preview scores remain qualified.']}
    try:
        from worldsync.released_reference_player import canonical_actions
        from worldsync.native_reference_timeline import NativeReferenceTimeline
        actions=canonical_actions(task)
        timeline=NativeReferenceTimeline(actions, task['prompt'])
        result['timeline']=timeline.quantization
        result.update(status='blocked_controller_integration',reference_actions=actions,
                      generation_blocker=GENERATION_BLOCKER)
    except (PlayerResponseError, ValueError) as error:
        result.update(status='unsupported_native_controls',error=str(error))
    paths=['worldsync/lingbot_released_baseline.py','worldsync/released_reference_player.py',
           'worldsync/lingbot_rollout.py','worldsync/lingbot_controls.py',
           'worldsync/native_reference_timeline.py','worldsync/camera_conditioning.py',
           'external/PlayWorld/Agent_player/hyworld2/player.py',
           f'external/PlayWorld/data/{split}/data.json']
    result['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    generation_argv(split,task_id,'episode',seed=seed,chunks=chunks)
    return result


def run(split,task_id,output,*,seed=42,chunks=60,validate_runtime=False):
    from worldsync import lingbot_rollout
    from worldsync.released_reference_player import ReleasedReferenceController
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    audit=preflight(split,task_id,seed=seed,chunks=chunks)
    if validate_runtime:
        if chunks > 2 or audit['status'] != 'blocked_controller_integration':
            raise ValueError('Runtime validation requires a supported plan and at most2chunks')
        audit.update(status='ready_for_generation_validation', scope='runtime smoke only; not benchmark evidence')
    (output/'protocol.json').write_text(json.dumps(audit,indent=2)+'\n')
    if audit['status']!='ready_for_generation_validation':
        raise ValueError('Native control preflight failed; task retained as unsupported')
    controller=ReleasedReferenceController();player=BaselinePlayer(controller,chunks)
    try:
        report=lingbot_rollout.main(generation_argv(split,task_id,output/'episode',seed=seed,chunks=chunks),
            player_decide=player.decide,simulation_chunk_budget=chunks, controller_profile=audit)
        if report['text_prompt_mode']!='prompt' or report['local_attn_size']!=18 or report['sink_size']!=6:
            raise ValueError('Corrected baseline settings changed')
        audit.update(generation_status=report['status'],haiku_requests=player.requests,
                     generated_seconds=report.get('generated_seconds'))
        return report
    except BaseException as error:
        audit.update(generation_status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (output/'protocol.json').write_text(json.dumps(audit,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['gc','if','insight','outsight'],required=True)
    parser.add_argument('--task-id',required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--chunks',type=int,default=60)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--generate',action='store_true',help='Explicitly run native GPU inference; omitted for CPU preflight')
    parser.add_argument('--validate-runtime',action='store_true',help='Permit at most2GPUchunks to validate rollout integration; no benchmark claim')
    args=parser.parse_args()
    if args.validate_runtime and not args.generate:
        parser.error('--validate-runtime requires --generate')
    if args.generate:
        run(args.split,args.task_id,args.output,seed=args.seed,chunks=args.chunks,validate_runtime=args.validate_runtime)
    else:
        result=preflight(args.split,args.task_id,seed=args.seed,chunks=args.chunks)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'profile':PROFILE,'task_id':args.task_id,'status':result['status']}))


if __name__=='__main__':
    main()
