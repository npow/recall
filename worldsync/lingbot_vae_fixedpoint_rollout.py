"""Opt-in research integration:637-frame conditioning, two generated chunks."""
import argparse
import hashlib
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--enable-shortcut',action='store_true')
    a=p.parse_args()
    import worldsync.lingbot_rollout as rollout
    import worldsync.lingbot_thor_smoke as smoke
    from worldsync.lingbot_vae_fixedpoint import VaeFixedPoint
    original_backend=rollout.configure_backend
    original_builder=rollout.build_pipeline
    original_main=rollout.main
    state={'encoder_rgb_groups':[],'conv1_shapes':[]}
    handles=[]
    shortcut=None
    def backend(report,save):
        state.update(report=report,save=save)
        report['runtime_validation']={'scope':'637-frame conditioning, intentionally stop after two rendered chunks',
            'benchmark_eligible':False,'declared_chunks':40,'rendered_chunk_limit':2,
            'vae_fixedpoint_enabled':a.enable_shortcut,
            'prototype_sha256':hashlib.sha256(Path(__file__).with_name('lingbot_vae_fixedpoint.py').read_bytes()).hexdigest()}
        save()
        return original_backend(report,save)
    def build(*values,**options):
        nonlocal shortcut
        pipeline=original_builder(*values,**options)
        shortcut=VaeFixedPoint(pipeline.vae.model,enabled=a.enable_shortcut)
        shortcut.__enter__()
        def encoder(module,positional):state['encoder_rgb_groups'].append(int(positional[0].shape[2]))
        def conv1(module,positional):state['conv1_shapes'].append(list(positional[0].shape))
        def stop(module,positional,keywords):
            report=state['report']
            if len(report['chunks'])>=2:
                report['termination']='intentional_runtime_validation_stop_after_two_chunks'
                state['save']()
                raise rollout.RolloutComplete
        handles.extend([pipeline.vae.model.encoder.register_forward_pre_hook(encoder),
                        pipeline.vae.model.conv1.register_forward_pre_hook(conv1),
                        pipeline.model.register_forward_pre_hook(stop,with_kwargs=True)])
        return pipeline
    def run(argv=None,runtime=None):
        values=list(sys.argv[1:] if argv is None else argv)
        values[values.index('--max-chunks')+1]='40'
        start=values.index('--fixed-actions')+1
        end=next(i for i in range(start,len(values)) if values[i].startswith('--'))
        actions=['WAIT']+['RIGHT']*12+['WAIT']*27
        values=values[:start]+actions+values[end:]
        try:return original_main(values,runtime=runtime)
        finally:
            for handle in handles:handle.remove()
            if shortcut is not None:shortcut.close()
            if 'report' in state:
                state['report']['runtime_validation'].update(
                    encoder_rgb_groups=state['encoder_rgb_groups'],conv1_shapes=state['conv1_shapes'],
                    instance_binding_restored=shortcut.closed if shortcut is not None else None,
                    shortcut=shortcut.metadata() if shortcut is not None else None)
                state['save']()
    try:
        rollout.configure_backend=backend
        rollout.build_pipeline=build
        rollout.main=run
        sys.argv=['lingbot_thor_smoke','--output',str(a.output),'--episode','baseline']
        smoke.main()
    finally:
        rollout.configure_backend=original_backend
        rollout.build_pipeline=original_builder
        rollout.main=original_main
        for handle in handles:handle.remove()
        if shortcut is not None:shortcut.close()

if __name__=='__main__':main()
