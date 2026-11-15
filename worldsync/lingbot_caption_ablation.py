"""Explicit experimental caption-plus-task arm; never a released baseline."""
import argparse
import hashlib
import json
from pathlib import Path

from worldsync import lingbot_released_baseline as baseline


def prepare(task_id, output, *, seed=42, chunks=60):
    audit = baseline.preflight('insight', task_id, seed=seed, chunks=chunks)
    if audit['status'] != 'blocked_controller_integration':
        raise ValueError('Caption ablation requires a supported canonical plan')
    task = baseline.load_task('insight', task_id)
    audit.update(profile='lingbot-insight-caption-ablation-v1',
                 experimental_text_mode='both',
                 world_text=f"{task['image_caption']} {task['prompt']}",
                 world_text_source='image_caption + space + task.prompt',
                 scope='experimental prompt ablation; not baseline reproduction',
                 status='ready_for_experimental_generation')
    audit['remaining_differences'] = [*audit['remaining_differences'],
        'Experimental caption-plus-task text differs from the released objective-only controller profile.']
    audit['source_sha256']['worldsync/lingbot_caption_ablation.py'] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    argv = baseline.generation_argv('insight', task_id, Path(output)/'episode', seed=seed, chunks=chunks)
    argv[argv.index('--text-prompt')+1] = 'both'
    argv.append('--controlled-ablation')
    return audit, argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--chunks', type=int, default=60)
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    audit, argv = prepare(args.task_id, args.output, seed=args.seed, chunks=args.chunks)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = args.output/'protocol.json'
    protocol.write_text(json.dumps(audit, indent=2)+'\n')
    if not args.generate:
        (args.output/'generation-argv.json').write_text(json.dumps(argv, indent=2)+'\n')
        return
    from worldsync import lingbot_rollout
    from worldsync.released_reference_player import ReleasedReferenceController
    player = baseline.BaselinePlayer(ReleasedReferenceController(), args.chunks)
    try:
        report = lingbot_rollout.main(argv, player_decide=player.decide,
            simulation_chunk_budget=args.chunks, controller_profile=audit)
        audit.update(generation_status=report['status'], haiku_requests=player.requests,
                     generated_seconds=report.get('generated_seconds'))
    except BaseException as error:
        audit.update(generation_status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        protocol.write_text(json.dumps(audit, indent=2)+'\n')


if __name__ == '__main__':
    main()
