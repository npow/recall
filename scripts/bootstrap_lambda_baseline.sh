#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/worldsync
python3 -m venv .venv-baseline
.venv-baseline/bin/python -m pip install -q uv
.venv-baseline/bin/uv pip install --python .venv-baseline/bin/python torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.venv-baseline/bin/uv pip install --python .venv-baseline/bin/python transformers==4.51.3 tokenizers==0.21.4 numpy==1.26.4 scipy==1.15.3 diffusers==0.37.0 huggingface_hub==0.36.2 hf_xet accelerate safetensors Pillow einops easydict ftfy sentencepiece tqdm imageio imageio-ffmpeg opencv-python-headless==4.11.0.86 anthropic==2.8.0
mkdir -p external/LingBot-World2
if [ ! -d external/LingBot-World2/.git ]; then
  git -C external/LingBot-World2 init
  git -C external/LingBot-World2 remote add origin https://github.com/Robbyant/lingbot-world-v2.git
  git -C external/LingBot-World2 fetch --depth 1 origin 1895d300d8ac936401689b26389f51cbd36530eb
  git -C external/LingBot-World2 checkout FETCH_HEAD
fi
HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DOWNLOAD_TIMEOUT=180 .venv-baseline/bin/python scripts/stage_lambda_baseline_weights.py
