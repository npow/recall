"""Check H100 eligibility before staging or launching model experiments."""
import modal

app = modal.App("worldsync-h100-availability")


@app.function(image=modal.Image.debian_slim(python_version="3.12"), gpu="H100", timeout=60)
def check():
    import subprocess
    return subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()


@app.local_entrypoint()
def main():
    print(check.remote())
