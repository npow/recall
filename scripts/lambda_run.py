"""Run a sharded LingBot case list on Lambda Cloud GPUs, copy results back, and terminate the instances.

    python scripts/lambda_run.py --ids-file outputs/lambda_ids.txt --label both --cases-file outputs/final_cases.txt \
        --out outputs/final/lambda-both -- --text-prompt both

The Lambda API key comes from `pass show api/lambda-cloud` and the Anthropic key from .env.local; neither is written to
the instances' disks (the Anthropic key is passed in the remote command's environment). Instances are always terminated:
when their shard finishes, on error, or at --max-hours.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
KEY = Path.home() / ".ssh/lambda_worldsync"
SSH = ["ssh", "-i", str(KEY), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
       "-o", "ServerAliveInterval=30", "-o", "LogLevel=ERROR"]
REMOTE = "/home/ubuntu/proj"
LOCK = threading.Lock()

SETUP = r"""
set -e
cd {remote}
sudo apt-get update -qq >/dev/null 2>&1 || true
sudo apt-get install -y -qq ffmpeg git >/dev/null 2>&1 || true
python3 -m pip install -q --user uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH
uv venv --python 3.12 venv -q
uv pip install -q --python venv/bin/python torch==2.6.0 torchvision==0.21.0
uv pip install -q --python venv/bin/python transformers==4.51.3 numpy==1.26.4 "tokenizers>=0.20.3" opencv-python-headless diffusers accelerate tqdm "imageio[ffmpeg]" easydict ftfy scipy omegaconf einops safetensors sentencepiece Pillow anthropic pydantic huggingface_hub hf_xet
HF_XET_HIGH_PERFORMANCE=1 venv/bin/python - <<'EOF'
import json
from pathlib import Path
from huggingface_hub import snapshot_download
target = Path("{remote}/external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast")
snapshot_download(repo_id="robbyant/lingbot-world-v2-14b-causal-fast", revision="5c33dd40b213598c418fd25bff30fdbd23fd38a7",
    local_dir=str(target), max_workers=16,
    allow_patterns=["transformers/*", "google/*", "config.json", "Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "README.md"])
manifest = json.loads(Path("{remote}/reproduction/lingbot-weights.json").read_text())
bad = [f["path"] for f in manifest["files"] if not (target / f["path"]).exists() or (target / f["path"]).stat().st_size != f["bytes"]]
assert not bad, f"bad weight files: {{bad}}"
print("weights verified", len(manifest["files"]))
EOF
if ! command -v ffmpeg >/dev/null; then
  mkdir -p $HOME/.local/bin
  ln -sf "$(venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')" $HOME/.local/bin/ffmpeg
fi
command -v ffmpeg >/dev/null || PATH=$HOME/.local/bin:$PATH command -v ffmpeg >/dev/null || {{ echo "ffmpeg missing" >&2; exit 3; }}
touch {remote}/SETUP_DONE
"""


def api(method, path, key, body=None):
    request = urllib.request.Request(f"https://cloud.lambdalabs.com/api/v1/{path}", method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": "Basic " + base64.b64encode(f"{key}:".encode()).decode(), "Content-Type": "application/json",
                 "User-Agent": "worldsync-lambda/1.0"})  # the default Python user agent is rejected with 403
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def run(command, **kwargs):
    return subprocess.run(command, capture_output=True, text=True, **kwargs)


def log(message):
    with LOCK:
        print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def anthropic_key():
    for line in (ROOT / ".env.local").read_text().splitlines():
        name, _, value = line.partition("=")
        if name == "ANTHROPIC_API_KEY":
            return value.strip()
    raise SystemExit("ANTHROPIC_API_KEY missing")


def worker(index, instance, shard, args, lambda_key, flags):
    ip, ident = instance["ip"], instance["id"]
    ssh = SSH + [f"ubuntu@{ip}"]
    out_local = args.out
    deadline = time.monotonic() + args.max_hours * 3600
    try:
        for _ in range(60):  # wait for sshd
            if run(ssh + ["true"]).returncode == 0:
                break
            time.sleep(10)
        else:
            raise RuntimeError("ssh never came up")
        log(f"[{index}] {ip} reachable; syncing code")
        run(ssh + [f"mkdir -p {REMOTE}/external/PlayWorld {REMOTE}/reproduction {REMOTE}/scripts {REMOTE}/out"])
        excludes = ["--exclude", "__pycache__", "--exclude", "checkpoints", "--exclude", ".venv*"]
        rsync = ["rsync", "-az", "-e", " ".join(SSH)] + excludes
        for src, dst in ((ROOT / "worldsync", REMOTE + "/"), (ROOT / "external/LingBot-World2", REMOTE + "/external/"),
                         (ROOT / "external/PlayWorld/data", REMOTE + "/external/PlayWorld/"),
                         (ROOT / "scripts/warm_suite.py", REMOTE + "/scripts/"),
                         (ROOT / "reproduction/lingbot-weights.json", REMOTE + "/reproduction/"),
                         (ROOT / "reproduction/state.json", REMOTE + "/reproduction/")):
            result = run(rsync + [str(src), f"ubuntu@{ip}:{dst}"])
            if result.returncode:
                raise RuntimeError(f"rsync failed {src}: {result.stderr[-200:]}")
        log(f"[{index}] setup (env + weights)")
        result = run(ssh + ["bash -s"], input=SETUP.format(remote=REMOTE))
        if result.returncode:
            raise RuntimeError(f"setup failed: {result.stderr[-400:]} {result.stdout[-300:]}")
        if args.setup_only:
            check = run(ssh + [f"cd {REMOTE} && export PATH=$HOME/.local/bin:$PATH && ffmpeg -version | head -1 && venv/bin/python -c 'import torch,transformers,cv2,anthropic;print(torch.__version__,torch.cuda.is_available())' && "
                               f"ls external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast | head -3 && du -sh external/LingBot-World2/checkpoints && df -h {REMOTE} | tail -1 && "
                               f"cd {REMOTE} && PYTHONPATH={REMOTE} venv/bin/python -m worldsync.lingbot_rollout --split gc --task-id GC007 --text-prompt both --offload-text-encoder --dry-run --output /tmp/dry && echo rollout-dry-run-ok"])
            log(f"[{index}] setup-only check:\n{check.stdout[-600:]}{check.stderr[-300:]}")
            return
        cases = " ".join(shard)
        # Write the run script without secrets; the key is only in the launching command's environment.
        script = (f"cd {REMOTE}\nexport PATH=$HOME/.local/bin:$PATH\nrm -f SHARD_DONE\nPYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH={REMOTE} "
                  f"venv/bin/python -u scripts/warm_suite.py --cases {cases} --out out/{args.label} --seed 42 -- --offload-text-encoder {flags} "
                  f"> run.log 2>&1\ntouch SHARD_DONE\n")
        run(ssh + [f"cat > {REMOTE}/run_shard.sh"], input=script)
        run(ssh + [f"ANTHROPIC_API_KEY='{anthropic_key()}' nohup bash {REMOTE}/run_shard.sh > /dev/null 2>&1 &"])
        log(f"[{index}] started {len(shard)} cases")
        while time.monotonic() < deadline:
            time.sleep(120)
            sync = run(["rsync", "-az", "-e", " ".join(SSH), "--include", "*/", "--include", "rollout.mp4", "--include", "run.json",
                        "--include", "summary.json", "--exclude", "*", f"ubuntu@{ip}:{REMOTE}/out/{args.label}/", str(out_local) + "/"])
            done = run(ssh + [f"test -f {REMOTE}/SHARD_DONE && echo done || echo running"]).stdout.strip()
            if done == "done":
                break
        run(["rsync", "-az", "-e", " ".join(SSH), "--include", "*/", "--include", "rollout.mp4", "--include", "run.json",
             "--include", "summary.json", "--exclude", "*", f"ubuntu@{ip}:{REMOTE}/out/{args.label}/", str(out_local) + "/"])
        log(f"[{index}] shard finished")
    except Exception as failure:
        log(f"[{index}] ERROR {failure}")
    finally:
        try:
            api("POST", "instance-operations/terminate", lambda_key, {"instance_ids": [ident]})
            log(f"[{index}] terminated {ident[:8]}")
        except Exception as failure:
            log(f"[{index}] TERMINATE FAILED {ident}: {failure}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids-file", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cases-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=5.0)
    parser.add_argument("--setup-only", action="store_true", help="sync, build the environment, download and verify weights, run checks, then terminate")
    parser.add_argument("flags", nargs="*")
    args = parser.parse_args()
    lambda_key = subprocess.run(["pass", "show", "api/lambda-cloud"], capture_output=True, text=True).stdout.splitlines()[0].strip()
    ids = [line.strip() for line in args.ids_file.read_text().split() if line.strip()]
    instances = []
    for _ in range(60):
        data = {i["id"]: i for i in api("GET", "instances", lambda_key)["data"]}
        instances = [data[i] for i in ids if i in data and data[i]["status"] == "active" and data[i].get("ip")]
        if len(instances) == len(ids):
            break
        time.sleep(15)
    log(f"{len(instances)}/{len(ids)} instances active")
    cases = args.cases_file.read_text().split()
    cases.sort(key=lambda c: not c.startswith("insight"))  # longest cases first, dealt round-robin
    shards = [cases[i::len(instances)] for i in range(len(instances))]
    args.out.mkdir(parents=True, exist_ok=True)
    threads = [threading.Thread(target=worker, args=(i, inst, shards[i], args, lambda_key, " ".join(args.flags)))
               for i, inst in enumerate(instances)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    leftovers = [i for i in ids if i in {x["id"] for x in api("GET", "instances", lambda_key)["data"]}]
    log(f"done; instances still listed: {len(leftovers)}")


if __name__ == "__main__":
    main()
