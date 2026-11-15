"""Can this Modal workspace launch a given GPU type?  PROBE_GPU=H100 modal run scripts/modal_gpu_probe.py"""
import os
import subprocess
import modal

GPU = os.environ.get("PROBE_GPU", "H100")
app = modal.App("worldsync-gpu-probe")
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, gpu=GPU, timeout=300)
def probe():
    return subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()


@app.local_entrypoint()
def main():
    print(GPU, "->", probe.remote(), flush=True)
