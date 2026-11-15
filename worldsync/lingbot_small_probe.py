"""Isolated official1.3B-release native replay; never changes14B defaults."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
REVISION='7e36a5f919f86cb4255cc9bfc30adb44963fbde1'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache-frames',type=int,choices=(18,70),required=True)
    a=p.parse_args()
    manifest=json.loads((ROOT/'reproduction/lingbot-small/weights.json').read_text())
    if manifest['checkpoint_revision']!=REVISION or manifest['status']!='verified':
        raise ValueError('Small checkpoint must be independently pinned and verified')
    small=Path(manifest['path'])
    for row in manifest['files']:
        if (small/row['path']).stat().st_size!=row['bytes']:
            raise ValueError('Small checkpoint inventory changed')
    import torch
    import worldsync.lingbot_rollout as rollout
    import worldsync.lingbot_thor_smoke as smoke
    original_backend=rollout.configure_backend
    original_builder=rollout.build_pipeline
    original_main=rollout.main
    original_schedule=rollout.SCHEDULES['stock']
    constructor_state={}

    def backend(report,save):
        upstream=original_backend(report,save)
        constructor_state['upstream']=upstream
        constructor_state['loader']=upstream.load_dit_model
        report.update(model='LingBot-World2 official1.3B-release causal-fast; independent small-model probe',
            protocol_id='lingbot-small-native-fixed13-v1',checkpoint_revision=REVISION,
            checkpoint_model_id=manifest['model_id'],asset_checkpoint_revision=rollout.WEIGHT_REVISION,
            model_weight_inventory=manifest['files'],
            small_probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            published_score_reproduced=False,
            deviations=['Separate small architecture/checkpoint, not a lossless14B optimization or leaderboard result',
                        'Native i2v-1.3B config; original four-step ODE with shift5',
                        'Reuse verified14B T5/VAE/tokenizer assets; discard T5 after one encoding',
                        'Single Thor, BF16, fused DiT SDPA; native VAE attention backend',
                        f'Declared fixed native-camera context experiment: cache{a.cache_frames}/sink6'])
        save()
        def strict_small(cls,checkpoint_dir,subfolder,config,torch_dtype,extra=None):
            if Path(checkpoint_dir).resolve()!=small.resolve() or torch_dtype!=torch.bfloat16:
                raise ValueError('Small probe attempted wrong checkpoint/dtype')
            kwargs=upstream._dit_kwargs_from_config(config,extra)
            model=cls(**kwargs)
            state=upstream._load_safetensors_state_dict(str(small))
            incompatible=model.load_state_dict(state,strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise ValueError('Small checkpoint did not strictly match official config')
            del state
            model=model.to(dtype=torch_dtype)
            report['checkpoint_loading']={'missing_keys':[],'unexpected_keys':[],
                'mismatched_keys':[],'error_msgs':[],
                'strategy':'official small-model config constructor and strict original sharded safetensors'}
            report['actual_dit_parameter_count']=sum(x.numel() for x in model.parameters())
            report['small_native_config']=kwargs
            save()
            return model
        upstream.load_dit_model=strict_small
        return upstream

    def build(upstream,configs,pack,knobs):
        return upstream.WanI2VCausal(configs['i2v-1.3B'],str(small),assets_dir=str(pack),
            local_attn_size=knobs['local_attn_size'],sink_size=knobs['sink_size'],
            infer_mode='causal_fast',convert_model_dtype=True)

    def run(argv=None,runtime=None):
        values=list(sys.argv[1:] if argv is None else argv)
        values+=['--controlled-ablation','--local-attn-size',str(a.cache_frames)]
        return original_main(values,runtime=runtime)

    try:
        rollout.configure_backend=backend
        rollout.build_pipeline=build
        rollout.main=run
        rollout.SCHEDULES['stock']={'timesteps_index':[0,250,500,750],'shift':5.0}
        sys.argv=['lingbot_thor_smoke','--output',str(a.output),'--episode','baseline']
        smoke.main()
    finally:
        rollout.configure_backend=original_backend
        rollout.build_pipeline=original_builder
        rollout.main=original_main
        rollout.SCHEDULES['stock']=original_schedule
        if constructor_state:
            constructor_state['upstream'].load_dit_model=constructor_state['loader']

if __name__=='__main__':
    main()
