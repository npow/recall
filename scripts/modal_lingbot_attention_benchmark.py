"""Compare stock SDPA, upgraded SDPA and FA4 on the same H100.

Torch 2.6 is kept in the original environment. FA4 uses a separate Torch 2.9
session because its current dependencies require newer dtypes and C++ ABI.
Neither environment changes the image or backend used by scored rollouts.
"""
import json
from pathlib import Path
import sys

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if modal.is_local():
    from scripts.modal_lingbot_benchmark import runtime_image
    fa4_runtime_image = (runtime_image
             .run_commands("python -m venv /opt/fa4",
                           "/opt/fa4/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128",
                           "/opt/fa4/bin/python -m pip install flash-attn-4==4.0.0b39 nvidia-cutlass-dsl==4.8.1 quack-kernels==0.6.5 apache-tvm-ffi==0.1.16 torch-c-dlpack-ext==0.1.5 numpy==1.26.4"))
    image = (fa4_runtime_image
             .add_local_file(ROOT / "worldsync/matrix_attention.py", "/work/matrix_attention.py")
             .add_local_file(ROOT / "scripts/lingbot_attention_kernel.py", "/work/attention_kernel.py"))
else:
    image = None

app = modal.App("worldsync-lingbot-attention-benchmark")


@app.function(image=image, gpu="H100", cpu=8, memory=16384, timeout=1200, max_containers=1)
def benchmark():
    import subprocess
    reports = {}
    for name, executable, flags in (("stock_torch26", sys.executable, []),
                                    ("candidate_torch29", "/opt/fa4/bin/python", ["--fa4"])):
        destination = Path("/tmp") / (name + ".json")
        outcome = subprocess.run([executable, "-u", "/work/attention_kernel.py", *flags,
                                  "--output", str(destination)], cwd="/work")
        if destination.exists():
            reports[name] = json.loads(destination.read_text())
        else:
            reports[name] = {"status": "failed", "returncode": outcome.returncode}
    return reports


@app.local_entrypoint()
def main():
    report = benchmark.remote()
    output = ROOT / "reproduction/lingbot-attention-kernel-benchmark.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all(row["status"] == "completed" for row in report.values()):
        raise RuntimeError("Kernel benchmark failed; diagnostic retained in " + str(output))
