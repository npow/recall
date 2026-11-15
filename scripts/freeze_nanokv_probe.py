"""Freeze the identity parity gate or subsequent compression diagnostic inputs."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--identity-proof', type=Path)
    args = parser.parse_args()
    run = json.loads((ROOT / 'reproduction/full-history-GC010/run.json').read_text())
    if run['status'] != 'completed' or len(run['chunks']) != 14 or run['local_attn_size'] != 58:
        raise ValueError('Native58 comparator incomplete')
    files = ['worldsync/' + n + '.py' for n in ['lingbot_nanokv', 'lingbot_nanokv_probe',
        'lingbot_rollout', 'lingbot_native', 'lingbot_controls', 'lingbot_paths', 'lingbot_vae',
        'lingbot_tuning', 'matrix_attention']]
    files += ['reproduction/lingbot-weights.json', 'reproduction/full-history-GC010/run.json',
              'reproduction/commit-forcing/GC010-seed42-h200/stock/run.json',
              'external/PlayWorld/data/gc/data.json']
    task = next(t for t in json.loads((ROOT / files[-1]).read_text()) if t['task_id'] == 'GC010')
    files.append('external/PlayWorld/data/' + task['image_path'])
    for n in ['lingbot_v2_cache_runtime.py', 'lingbot_v2_kv_codec.py', 'lingbot_v2_kv_codec_triton.py']:
        files.append('external/NanoKV/models/LingBot-World-2.0/compression/wan/modules/' + n)
    files += ['external/LingBot-World2/wan/modules/model_fast.py', 'external/LingBot-World2/wan/image2video.py']
    plan = {'protocol': 'nanokv-isolated-gc010-v1', 'benchmark_eligible': False,
            'task_id': 'GC010', 'seed': 42, 'max_chunks': 14, 'actions': run['fixed_actions'],
            'turn_degrees_per_second': run['turn_degrees_per_second'],
            'initial_image_sha256': run['initial_image_sha256'],
            'image_caption_sha256': run['image_caption_sha256'], 'text_prompt_mode': 'caption',
            'stock18_pixel_hashes': [r['pixel_sha256'] for r in json.loads((ROOT / 'reproduction/commit-forcing/GC010-seed42-h200/stock/run.json').read_text())['chunks']],
            'native58_pixel_hashes': [r['pixel_sha256'] for r in run['chunks']],
            'codec_sha256': 'b014909923f657b8019203bd581da77e8c6e0880bf7aa80b3eadfbf6a433f458',
            'codec_hf_revision': '4b0dac1219ed72ead539f8d063348949e6aeb045',
            'nanokv_source_revision': 'a9baa538263e1c19acc05d10df80503c521c803c',
            'source_revision': run['source_revision'], 'checkpoint_revision': run['checkpoint_revision'],
            'hardware': run['hardware'], 'attention_backend': 'sdpa',
            'sampler_timesteps_index': run['sampler_timesteps_index'], 'sample_shift': run['sample_shift'],
            'identity_gpu_gate_passed': False, 'input_files': {name: sha(ROOT / name) for name in files},
            'decision': 'First require all14 identity58 chunk hashes exact. Then compare codec18 with stock18 and codec58 with native58. Retain all failures and inspect actual returns; no overall claim.'}
    if args.identity_proof:
        proof = json.loads(args.identity_proof.read_text())
        summary = proof['summary']
        equal = summary.get('raw_pixel_equal_native58')
        if (proof['returncode'] != 0 or summary['status'] != 'completed'
                or equal != [True]*14 or not summary.get('base_parameter_versions_unchanged')
                or not summary.get('adapter', {}).get('identity')
                or not summary.get('adapter', {}).get('closed')
                or summary.get('input_files') != plan['input_files']):
            raise ValueError('Identity GPU proof has not passed')
        plan['identity_gpu_gate_passed'] = True
        plan['identity_gpu_proof'] = {'path': str(args.identity_proof), 'sha256': sha(args.identity_proof),
            'result_directory': proof['result_directory'], 'all14_pixels_exact': True,
            'frozen_parameters': True, 'bindings_closed': True}
    plan['plan_sha256'] = hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if args.output.exists():
        raise FileExistsError('Frozen plan may not be overwritten')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(plan,indent=2)+'\n')
    print(plan['plan_sha256'])


if __name__ == '__main__':
    main()
