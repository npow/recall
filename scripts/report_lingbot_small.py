"""Package complete native small-model baseline/context evidence without scoring it."""
import argparse
import json
from pathlib import Path
import shutil
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline',type=Path)
    p.add_argument('long_context',type=Path)
    p.add_argument('--output',type=Path,default=Path('reproduction/lingbot-small'))
    a=p.parse_args()
    rows={}
    runs={}
    for name,source in [('cache18',a.baseline),('cache70',a.long_context)]:
        dest=a.output/name
        dest.mkdir(parents=True,exist_ok=True)
        for file in ['result.json','supervisor.json','memory-samples.jsonl','supervisor-samples.jsonl','child.log']:
            if (source/file).exists():shutil.copy2(source/file,dest/file)
        for file in ['run.json','rollout.mp4']:
            if (source/'rollout'/file).exists():shutil.copy2(source/'rollout'/file,dest/file)
        result=json.loads((source/'result.json').read_text())
        run=result['run']
        runs[name]=run
        chunks=run['chunks']
        intervals=[chunks[i]['elapsed_s']-chunks[i-1]['elapsed_s'] for i in range(9,13)]
        rows[name]={'status':run['status'],'chunks':len(chunks),'generated_frames':run['generated_frames'],
            'harness_total_s':result['elapsed_s'],'native_gpu_peak_bytes':run['peak_allocated_bytes'],
            'sampled_reserved_peak_bytes':result['peak_sampled_cuda_reserved_bytes'],
            'min_physical_available_bytes':result['min_mem_available_bytes'],
            'first_chunk_s':chunks[0]['elapsed_s'],'last_four_intervals_s':intervals,
            'last_four_mean_s':statistics.mean(intervals),
            'native_camera_audits_pass':len(run['camera_embedding_audit'])==13 and all(x['within_tolerance'] for x in run['camera_embedding_audit'])}
    left,right=runs['cache18'],runs['cache70']
    fields=['checkpoint_revision','asset_checkpoint_revision','checkpoint_model_id','source_revision','initial_image_sha256','image_caption_sha256','task_id','split','seed','max_chunks','sampler_timesteps_index','sample_shift','fps','turn_degrees_per_second','fixed_actions','hardware','attention_backend','sink_size','actual_dit_parameter_count','adapter_sha256','small_probe_sha256']
    matches={key:left.get(key)==right.get(key) for key in fields}
    hashes=[x['pixel_sha256']==y['pixel_sha256'] for x,y in zip(left['chunks'],right['chunks'])]
    report={'scope':'Independent1.3B-release speed/quality context replay; not14B lossless optimization or leaderboard evidence',
            'actual_dit_parameter_count':left['actual_dit_parameter_count'],'runs':rows,
            'matched_settings':matches,'declared_changed_setting':{'local_attn_size':[left['local_attn_size'],right['local_attn_size']]},
            'chunk_pixel_hash_equal':hashes,'first_differing_chunk':next((i for i,v in enumerate(hashes) if not v),None),
            'caveats':['Different model and official shift from14B: speed/quality trade-off, not transparent optimization.',
                       'Cold loading varied substantially; compare warmed intervals rather than total times.',
                       'Thirteen fixed native-camera chunks test one round trip, not adaptive PlayWorld or all four dimensions.',
                       'No task score or quality improvement inferred from longer context.']}
    (a.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not(all(matches.values()) and all(x['status']=='completed' and x['chunks']==13 and x['native_camera_audits_pass'] for x in rows.values()) and all(hashes[:4])):
        raise RuntimeError('Small-model pair incomplete or unmatched; artifacts preserved')

if __name__=='__main__':main()
