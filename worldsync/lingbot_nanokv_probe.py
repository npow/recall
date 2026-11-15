"""Isolated frozen NanoKV diagnostic; never changes the default rollout."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--arm', choices=['identity58', 'codec18', 'codec58'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--codec', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    body = {k: v for k, v in plan.items() if k != 'plan_sha256'}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != plan['plan_sha256']:
        raise ValueError('Plan changed')
    for name, value in plan['input_files'].items():
        if sha(ROOT / name) != value:
            raise ValueError('Frozen input changed: ' + name)
    if sha(args.codec) != plan['codec_sha256']:
        raise ValueError('Codec changed')
    if args.arm != 'identity58' and not plan.get('identity_gpu_gate_passed', False):
        raise ValueError('Identity GPU parity gate must pass before compressed arms')
    capacity = 18 if args.arm == 'codec18' else 58
    if plan['task_id'] != 'GC010' or len(plan['actions']) != 14 or plan['seed'] != 42:
        raise ValueError('Unexpected frozen native diagnostic')
    args.output.mkdir(parents=True, exist_ok=False)
    from worldsync import lingbot_rollout as rollout
    from worldsync.lingbot_nanokv import NativeNanoKV
    original_build = rollout.build_pipeline
    state = {}
    summary = {'scope': 'Single scripted development-case diagnostic; not full PlayWorld overall',
               'arm': args.arm, 'plan_sha256': plan['plan_sha256'], 'status': 'running',
               'codec_sha256': plan['codec_sha256'], 'logical_capacity_frames': capacity,
               'input_files': plan['input_files']}

    def save():
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    def build(*positional, **kwargs):
        pipeline = original_build(*positional, **kwargs)
        state['pipeline'] = pipeline
        state['versions'] = {name: p._version for name, p in pipeline.model.named_parameters()}
        adapter = NativeNanoKV(pipeline, source_root=ROOT / 'external/NanoKV',
                              codec_path=args.codec, capacity_frames=capacity,
                              identity=args.arm == 'identity58')
        state['adapter'] = adapter
        adapter.__enter__()
        return pipeline

    started = time.monotonic()
    rollout.build_pipeline = build
    save()
    try:
        report = rollout.main(['--split', 'gc', '--task-id', 'GC010', '--seed', '42',
            '--max-chunks', '14', '--turn-degrees-per-second', str(plan['turn_degrees_per_second']),
            '--native-fixed-controls', '--controlled-ablation', '--local-attn-size', str(capacity),
            '--sink-size', '6', '--offload-text-encoder', '--fixed-actions', *plan['actions'],
            '--output', str(args.output / 'episode')])
        summary['raw_pixel_equal_native58'] = [r['pixel_sha256'] == expected
            for r, expected in zip(report['chunks'], plan['native58_pixel_hashes'])]
        summary['base_parameter_versions_unchanged'] = all(
            p._version == state['versions'][name] for name, p in state['pipeline'].model.named_parameters())
        if not summary['base_parameter_versions_unchanged']:
            raise ValueError('Frozen backbone parameters changed')
        if len(report['chunks']) != 14 or not all(r['within_tolerance'] for r in report['camera_embedding_audit']):
            raise ValueError('Incomplete run or native camera audit failed')
        if args.arm == 'identity58' and not all(summary['raw_pixel_equal_native58']):
            raise ValueError('Identity adapter did not reproduce every native58 chunk')
        summary['status'] = 'completed'
    except BaseException:
        summary['status'] = 'failed'
        summary['error'] = traceback.format_exc()
        raise
    finally:
        rollout.build_pipeline = original_build
        adapter = state.get('adapter')
        if adapter is not None:
            adapter.close()
            summary['adapter'] = adapter.metadata()
        run_path = args.output / 'episode/run.json'
        if run_path.exists():
            run = json.loads(run_path.read_text())
            run['nanokv_diagnostic'] = {'arm': args.arm, 'plan_sha256': plan['plan_sha256'],
                'identity': args.arm == 'identity58', 'codec_sha256': plan['codec_sha256'],
                'logical_capacity_frames': capacity, 'proof_status': summary['status']}
            run['model'] = 'LingBot-World2 14B causal-fast + isolated NanoKV diagnostic'
            run_path.write_text(json.dumps(run, indent=2) + '\n')
        summary['wall_s'] = time.monotonic() - started
        save()
        print(json.dumps({k: summary[k] for k in ['arm', 'status', 'wall_s']}), flush=True)


if __name__ == '__main__':
    main()
