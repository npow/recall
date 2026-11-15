"""Preserve guarded local pair evidence and check exact prefix before intervention."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument('baseline', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('--output', type=Path, default=Path('reproduction/lingbot-local-memory/file-archive-pair'))
    a = p.parse_args()
    report = {'scope': 'Local Thor full13chunk lossless-file4 transport pilot; no score claim', 'runs': {}}
    runs = {}
    for arm, source in [('baseline', a.baseline), ('file4', a.candidate)]:
        destination = a.output / arm
        destination.mkdir(parents=True, exist_ok=True)
        for name in ['result.json', 'supervisor.json', 'memory-samples.jsonl', 'supervisor-samples.jsonl', 'child.log']:
            if (source / name).exists():
                shutil.copy2(source / name, destination / name)
        for name in ['run.json', 'rollout.mp4']:
            if (source / 'rollout' / name).exists():
                shutil.copy2(source / 'rollout' / name, destination / name)
        result = json.loads((source / 'result.json').read_text()) if (source / 'result.json').exists() else {'status': 'child_no_final_result', 'min_mem_available_bytes': 0}
        run = result.get('run') or (json.loads((source / 'rollout/run.json').read_text()) if (source / 'rollout/run.json').exists() else {})
        runs[arm] = run
        report['runs'][arm] = {k: v for k, v in result.items() if k not in ['run', 'events']}
        report['runs'][arm]['supervisor'] = json.loads((source / 'supervisor.json').read_text())
    candidate, baseline = runs['file4'], runs['baseline']
    report['complete_13_chunks'] = all(r.get('status') == 'completed' and len(r.get('chunks', [])) == 13 for r in runs.values())
    fields = ['source_revision', 'checkpoint_revision', 'initial_image_sha256', 'image_caption_sha256', 'task_id', 'split', 'seed', 'turn_degrees_per_second', 'fps', 'max_chunks', 'fixed_actions', 'hardware', 'attention_backend', 'sampler_timesteps_index', 'sample_shift', 'local_attn_size', 'sink_size']
    report['native_camera_audits_pass'] = all(len(r.get('camera_embedding_audit', [])) == 13 and all(x['within_tolerance'] for x in r['camera_embedding_audit']) for r in runs.values())
    report['reserve_preserved'] = all(row['supervisor']['status'] == 'completed' and row['min_mem_available_bytes'] >= 16 * 1024 ** 3 for row in report['runs'].values())
    report['matched_settings'] = {key: baseline.get(key) == candidate.get(key) for key in fields}
    trace = candidate.get('memory', {}).get('trace', [])
    first = next((x['chunk'] for x in trace if x['attention_replaced']), None)
    report['first_attention_intervention_chunk'] = first
    hashes = [x['pixel_sha256'] == y['pixel_sha256'] for x,y in zip(baseline.get('chunks',[]),candidate.get('chunks',[]))]
    report['all_chunk_pixel_hash_equal'] = hashes
    report['all_pre_intervention_hashes_exact'] = first is not None and len(hashes) == 13 and all(hashes[:first])
    report['adapter_hash_equal'] = baseline.get('adapter_sha256') == candidate.get('adapter_sha256')
    report['caveat'] = 'Local Torch2.15 is a separate runtime from H200Torch2.6. Prefix audit checks transport isolation; last-chunk difference is an intervention, not evidence of improvement.'
    (a.output / 'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key != 'runs'},indent=2))
    if not(report['complete_13_chunks'] and all(report['matched_settings'].values()) and report['all_pre_intervention_hashes_exact'] and report['reserve_preserved'] and report['native_camera_audits_pass']):
        raise RuntimeError('Incomplete or unmatched local pair; retained artifacts require inspection')

if __name__ == '__main__':
    main()
