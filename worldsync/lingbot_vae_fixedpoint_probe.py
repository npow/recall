"""Standalone real-VAE exactness/timing probe; no DiT, T5, or generator edits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    import torch
    from worldsync.lingbot_vae_fixedpoint import encode_zero_tail
    from worldsync.lingbot_thor_smoke import meminfo,rss
    from worldsync.lingbot_native import SOURCE_REVISION
    import subprocess
    source=ROOT/'external/LingBot-World2'
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if revision!=SOURCE_REVISION:raise ValueError('Wrong pinned source')
    sys.path.insert(0,str(source))
    from wan.modules.vae2_1 import Wan2_1_VAE
    import torchvision.transforms.functional as TF
    from PIL import Image
    weights=ROOT/'external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast/Wan2.1_VAE.pth'
    manifest=json.loads((ROOT/'reproduction/lingbot-weights.json').read_text())
    record=next(v for v in manifest['files'] if v['path']=='Wan2.1_VAE.pth')
    h=hashlib.sha256()
    with weights.open('rb') as f:
        while data:=f.read(8*1024*1024):h.update(data)
    if h.hexdigest()!=record['sha256']:raise ValueError('VAE checksum differs from verified asset')
    started=time.monotonic()
    phase={'name':'loading'}
    samples=[]
    stop=threading.Event()
    report={'scope':'Standalone actual VAE positive-zero-tail fixed-point proof; inference only',
            'status':'running','source_revision':revision,'vae_sha256':h.hexdigest(),
            'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
            'prototype_sha256':hashlib.sha256(Path(__file__).with_name('lingbot_vae_fixedpoint.py').read_bytes()).hexdigest(),
            'default_generator_changed':False,'rows':[]}
    def sample():
        info=meminfo()
        row={'elapsed_s':time.monotonic()-started,'phase':phase['name'],'rss_bytes':rss(),
             'mem_available_bytes':info['MemAvailable'],'dirty_bytes':info.get('Dirty',0),
             'writeback_bytes':info.get('Writeback',0),
             'cuda_allocated_bytes':torch.cuda.memory_allocated(),'cuda_reserved_bytes':torch.cuda.memory_reserved()}
        samples.append(row)
        with (a.output/'memory-samples.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    def watch():
        while not stop.wait(.25):sample()
    watcher=threading.Thread(target=watch,daemon=True)
    watcher.start()
    def save():(a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        vae=Wan2_1_VAE(vae_pth=str(weights),device='cuda')
        versions={n:v._version for n,v in vae.model.named_parameters()}
        data_root=ROOT/'external/PlayWorld/data'
        task=next(x for x in json.loads((data_root/'gc/data.json').read_text()) if x['task_id']=='GC027')
        image_path=data_root/task['image_path']
        image=TF.to_tensor(Image.open(image_path).convert('RGB')).sub_(.5).div_(.5)
        first=torch.nn.functional.interpolate(image[None],size=(464,832),mode='bicubic')[0].cuda()
        report.update(native_vae_dtype=str(vae.dtype),source_image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                      input_height=464,input_width=832)
        save()
        with torch.inference_mode():
            phase['name']='warmup13'
            warm=torch.zeros(3,13,464,832,device='cuda');warm[:,0]=first
            vae.encode([warm]);del warm
            torch.cuda.synchronize()
            for frames in (205,637):
                x=torch.zeros(3,frames,464,832,device='cuda');x[:,0]=first
                torch.cuda.synchronize()
                outputs={}
                for mode in ('native','fixedpoint'):
                    phase['name']=f'{mode}_{frames}'
                    stats={}
                    original=vae.model.encode
                    if mode=='fixedpoint':
                        vae.model.encode=lambda tensor,scale:encode_zero_tail(vae.model,tensor,scale,original=original,stats=stats)
                    torch.cuda.reset_peak_memory_stats()
                    begin=time.monotonic()
                    try:
                        latent=vae.encode([x])[0]
                        torch.cuda.synchronize()
                    finally:vae.model.encode=original
                    elapsed=time.monotonic()-begin
                    finite=bool(torch.isfinite(latent).all())
                    cpu=latent.cpu().contiguous()
                    digest=hashlib.sha256(cpu.numpy().tobytes()).hexdigest()
                    outputs[mode]=cpu
                    report['rows'].append({'mode':mode,'rgb_frames':frames,'elapsed_s':elapsed,
                        'latent_shape':list(cpu.shape),'dtype':str(cpu.dtype),'sha256':digest,'finite':finite,
                        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                        'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'fixedpoint':stats})
                    save()
                    del latent
                equal=torch.equal(outputs['native'].view(torch.uint8),outputs['fixedpoint'].view(torch.uint8))
                report.setdefault('parity',[]).append({'rgb_frames':frames,'all_normalized_latent_bytes_exact':equal})
                save()
                if not equal or not all(row['finite'] for row in report['rows']):raise ValueError('Actual VAE output is not finite/bit-exact')
                del outputs,x
            report['parameter_versions_unchanged']=versions=={n:v._version for n,v in vae.model.named_parameters()}
            if not report['parameter_versions_unchanged']:raise ValueError('VAE parameters mutated')
        report['status']='completed'
    except BaseException:
        report.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        stop.set();watcher.join(timeout=2);sample()
        report.update(elapsed_s=time.monotonic()-started,min_mem_available_bytes=min(v['mem_available_bytes'] for v in samples),
                      peak_rss_bytes=max(v['rss_bytes'] for v in samples))
        save()
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
