"""One opt-in B200 FA4 trial, preserving the matched H100 ten-WAIT protocol."""
import json
from pathlib import Path
import statistics
import sys
import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.modal_lingbot_runtime_benchmark import image, weights, results, generate
app = modal.App('worldsync-lingbot-b200-runtime')

@app.function(image=image, gpu='B200', cpu=8, memory=131072, timeout=1800,
              volumes={'/cache': weights, '/results': results})
def trial():
    import os
    import subprocess
    import time
    start = time.monotonic()
    code = '''import json, torch
from tvm_ffi._optional_torch_c_dlpack import patch_torch_cuda_stream_protocol
patch_torch_cuda_stream_protocol()
from flash_attn.cute import flash_attn_func
q=torch.randn(1,128,40,128,device="cuda",dtype=torch.bfloat16)
y=flash_attn_func(q,q,q)
assert torch.isfinite(y).all()
print(json.dumps({"gpu":torch.cuda.get_device_name(),"capability":torch.cuda.get_device_capability(),"torch":torch.__version__,"cuda":torch.version.cuda,"finite":True}))
'''
    check = subprocess.run(['/opt/fa4/bin/python', '-c', code], capture_output=True, text=True,
                           env=dict(os.environ, TVM_FFI_DISABLE_TORCH_C_DLPACK='1'))
    report = {'preflight_wall_s': time.monotonic()-start, 'preflight_returncode': check.returncode,
              'preflight_stdout': check.stdout, 'preflight_stderr': check.stderr}
    if check.returncode:
        return dict(report, status='preflight_failed')
    result = generate.local('fa4', True)
    report.update(result)
    report['worker_total_wall_s'] = time.monotonic()-start
    return report

@app.local_entrypoint()
def main():
    output = ROOT / 'reproduction/lingbot-runtime-benchmark'
    try:
        report = trial.remote()
    except Exception as error:
        (output/'b200-launch-failure.json').write_text(json.dumps({'status':'launch_failed','error':str(error)},indent=2)+'\n')
        raise
    (output/'b200-fa4.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['status'] != 'completed':
        raise RuntimeError('B200 preflight or generation failed; artifacts retained, no retry')
    run = report['run']
    reference = json.loads((output/'fa4-no-addon.json').read_text())['run']
    intervals = [run['chunks'][i]['elapsed_s']-run['chunks'][i-1]['elapsed_s'] for i in range(6,10)]
    old = [reference['chunks'][i]['elapsed_s']-reference['chunks'][i-1]['elapsed_s'] for i in range(6,10)]
    summary = {'scope':'ten native WAIT chunks; opt-in kernel/runtime throughput, no task-quality claim',
               'hardware':run['hardware'], 'warmed_mean_s':statistics.mean(intervals),
               'warmed_intervals_s':intervals,'h100_warmed_mean_s':statistics.mean(old),
               'speedup_vs_h100_fa4':statistics.mean(old)/statistics.mean(intervals),
               'first_chunk_elapsed_s':run['chunks'][0]['elapsed_s'],
               'peak_allocated_bytes':run['peak_allocated_bytes'],
               'native_camera_audit_passed':len(run['camera_embedding_audit'])==10 and all(x['within_tolerance'] for x in run['camera_embedding_audit']),
               'caveat':'Different GPU architecture; no exact BF16 pixel equivalence assumed. Small-shape kernel preflight precedes weight loading.'}
    (output/'b200-comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
