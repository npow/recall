"""Resume only missing full1 judgments without mutating the collaborator's worktree."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worldsync.gemini_vertex_judge import load_dataset_task, run_manifest


def main():
    source = ROOT / '.claude/worktrees/agent-a20dd1a4d88b1a928/outputs/adaptive/full1'
    output = ROOT / 'reproduction/full1-recovered-judging'
    evidence = output / 'episodes'
    evidence.mkdir(parents=True, exist_ok=True)
    plan = json.loads((source / 'plan.json').read_text())
    shutil.copy2(source / 'plan.json', output / 'generation-plan.json')
    expected = {f'{arm}-{case.split(":")[0]}-{case.split(":")[1]}-seed{seed}'
                for arm in plan['arms'] for case in plan['cases'] for seed in plan['seeds']}
    progress = {}
    for path in (ROOT / 'outputs/full1-status/progress-records').rglob('*.json'):
        row = json.loads(path.read_text())
        identifier = f"{row['arm']}-{row['split']}-{row['task_id']}-seed{row['seed']}"
        if identifier in progress:
            raise ValueError('Duplicate generation record: ' + identifier)
        if row['status'] == 'skipped_already_complete':
            if identifier != 'stock-if-IF030-seed42':
                raise ValueError('Unverified skipped run: ' + identifier)
            run_path = ROOT / 'outputs/full1-status/skipped-if030/run.json'
            verified = json.loads(run_path.read_text())
            if any(verified.get(k) != v for k, v in {
                    'status': 'completed', 'task_id': row['task_id'],
                    'split': row['split'], 'seed': row['seed']}.items()):
                raise ValueError('Skipped run does not match completed manifest')
            row = dict(row, status='completed', original_progress_status='skipped_already_complete',
                       result_directory=row['base'] + '__a0',
                       completion_verified_from_run_sha256=hashlib.sha256(run_path.read_bytes()).hexdigest())
            shutil.copy2(run_path, output / 'skipped-if030-run.json')
        progress[identifier] = row
    if set(progress) != expected or len(expected) != 342:
        raise ValueError(f'Incomplete frozen generation matrix: {len(progress)}/{len(expected)}')
    if any(r['status'] != 'completed' for r in progress.values()):
        raise ValueError('Generation failures need explicit released failure-policy handling')
    (output / 'generation-progress.json').write_text(json.dumps(progress, indent=2) + '\n')
    copied = []
    pending = []
    for identifier, row in sorted(progress.items()):
        target = evidence / f'{identifier}.json'
        previous = source / 'gemini/episodes' / f'{identifier}.json'
        if not target.exists() and previous.exists():
            judgment = json.loads(previous.read_text())
            video = source / 'episodes' / identifier / 'rollout.mp4'
            if (judgment.get('scores', {}).get('official') is not None
                    and video.exists() and hashlib.sha256(video.read_bytes()).hexdigest() == judgment['video_sha256']):
                shutil.copy2(previous, target)
                copied.append(identifier)
        if target.exists() and json.loads(target.read_text()).get('scores', {}).get('official') is not None:
            continue
        pending.append((identifier, row))
    (output / 'recovery.json').write_text(json.dumps({
        'scope': 'Full342 two-arm generated matrix, same released judge and neutral model label; existing judgments preserved',
        'copied_verified_judgments': copied, 'pending_at_start': [i for i, _ in pending],
        'credential_recovery_project': 'servo-tpu-20260901',
        'caveat': 'Local integrated LingBot with streamed Haiku player and Gemini preview alias; not a reproduction of the hosted LingBot service.'}, indent=2) + '\n')

    def prepare(item):
        identifier, row = item
        local = source / 'episodes' / identifier / 'rollout.mp4'
        if not local.exists():
            local = ROOT / 'outputs/full1-recovery-videos' / identifier / 'rollout.mp4'
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists():
                subprocess.run([str(ROOT / '.venv-modal-new/bin/modal'), 'volume', 'get', '--profile', 'nissan-pow',
                                'worldsync-lingbot-results', row['result_directory'] + '/rollout.mp4', str(local), '--force'],
                               check=True, capture_output=True, text=True)
        task = load_dataset_task(ROOT / 'external/PlayWorld', row['split'], row['task_id'])
        return {'episode_id': identifier, 'split': row['split'], 'task_id': row['task_id'],
                'video': str(local.resolve()),
                'reference_image': str((ROOT / 'external/PlayWorld/data' / task['image_path']).resolve()),
                'world_model': 'LingBot-World2 14B causal-fast'}

    with ThreadPoolExecutor(max_workers=6) as pool:
        manifest_rows = list(pool.map(prepare, pending))
    manifest = output / 'pending-manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in manifest_rows))
    print(json.dumps({'expected': len(expected), 'preserved': len(expected)-len(pending), 'pending': len(pending)}), flush=True)
    if pending:
        run_manifest(manifest, output, workers=3, project='servo-tpu-20260901')


if __name__ == '__main__':
    main()
