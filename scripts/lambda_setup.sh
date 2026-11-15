#!/usr/bin/env bash
# Reproducible environment for a Lambda Cloud GPU instance (run ON the instance):
#   python scripts/lambda_fleet.py setup <ip>     (pushes the code, then runs this over SSH)
# Mirrors the Modal image in scripts/modal_lingbot_benchmark.py (Python 3.12, torch 2.6.0 cu124, same pins).
# Layout: ~/work is the repository root (work_root); ~/work/results holds episode artifacts.
set -euo pipefail
SOURCE_REVISION=1895d300d8ac936401689b26389f51cbd36530eb
WORK="$HOME/work"

if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"
sudo apt-get install -y -qq ffmpeg libgl1 libglib2.0-0 tmux rsync >/dev/null 2>&1 || true

# The repository is private, so the code is pushed by scripts/lambda_fleet.py (rsync of the origin/main checkout),
# not cloned here; ~/work/COMMIT records the commit.
test -d "$WORK/worldsync" || { echo "push the code first: scripts/lambda_fleet.py push"; exit 1; }
cat "$WORK/COMMIT"

LING="$WORK/external/LingBot-World2"
if [ ! -d "$LING/.git" ]; then
  mkdir -p "$LING"; git -C "$LING" init -q
  git -C "$LING" remote add origin https://github.com/Robbyant/lingbot-world-v2.git
  git -C "$LING" fetch -q --depth 1 origin "$SOURCE_REVISION"
  git -C "$LING" checkout -q FETCH_HEAD
fi

cd "$WORK"
[ -d .venv-lambda ] || uv venv -q --python 3.12 .venv-lambda
uv pip install -q --python .venv-lambda/bin/python torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -q --python .venv-lambda/bin/python transformers==4.51.3 tokenizers==0.21.4 numpy==1.26.4 scipy==1.15.3 \
  diffusers==0.37.0 huggingface_hub==0.36.2 hf_xet accelerate safetensors Pillow einops easydict ftfy sentencepiece tqdm \
  imageio imageio-ffmpeg opencv-python-headless==4.11.0.86 anthropic==2.8.0 pydantic requests
mkdir -p results outputs
.venv-lambda/bin/python - <<'PY'
import torch
print("torch", torch.__version__, torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GiB")
PY
echo SETUP_OK
