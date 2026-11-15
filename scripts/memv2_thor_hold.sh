#!/bin/bash
# Same-hardware (local Thor) runs on the full GC007 hold path: v2 recompute K=10, stock, old rule via recompute.
# No cloud cost. Outputs under outputs/thor-step1/ (ignored by git).
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/npow/code/iteratehack/.venv-lingbot/bin/python
ACTIONS="WAIT RIGHT RIGHT RIGHT WAIT WAIT WAIT WAIT WAIT WAIT LEFT LEFT LEFT WAIT"
COMMON="--split gc --task-id GC007 --seed 42 --max-chunks 14 --native-fixed-controls --fixed-actions $ACTIONS --turn-degrees-per-second 68 --offload-text-encoder"
mkdir -p outputs/thor-step1
$PY -u -m worldsync.lingbot_rollout $COMMON --memory-v2 recompute --memv2-select verified --output outputs/thor-step1/v2recompute-hold-GC007-seed42 > outputs/thor-step1/v2recompute.log 2>&1
$PY -u -m worldsync.lingbot_rollout $COMMON --output outputs/thor-step1/stock-hold-GC007-seed42 > outputs/thor-step1/stock.log 2>&1
$PY -u -m worldsync.lingbot_rollout $COMMON --memory-v2 recompute --memv2-select pose --output outputs/thor-step1/oldpose-hold-GC007-seed42 > outputs/thor-step1/oldpose.log 2>&1
