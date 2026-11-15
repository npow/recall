#!/usr/bin/env bash
# One status snapshot of the final run (run LOCALLY): counts, failures, per-instance worker state.
#   scripts/lambda_report.sh DIRECTOR_WORKTREE IP [IP ...]
set -uo pipefail
D="$1"; shift
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
date +%T
python3 scripts/lambda_final.py status --jobs "$D/reproduction/final-run-jobs.jsonl" --dest "$D/outputs/final1" --failed-dir "$D/outputs/final1-failed"
for ip in "$@"; do
  ssh -i "${LAMBDA_SSH_KEY:-$HOME/.ssh/id_rsa}" -o ConnectTimeout=15 "ubuntu@$ip" \
    'cd ~/final 2>/dev/null || { echo "'"$ip"' no final tree yet"; exit 0; }
     done=$(grep "\"event\": \"job\"" worker.log | grep -c "\"status\": \"completed\""); bad=$(grep "\"event\": \"job\"" worker.log | grep -vc "\"status\": \"completed\"")
     left=$(( $(wc -l < jobs.jsonl) - done ))
     state=$(tmux has-session -t worker 2>/dev/null && echo running || echo STOPPED)
     echo "'"$ip"' done=$done failed_attempts=$bad left=$left worker=$state gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader) restarts=$(grep -c worker-exited-nonzero worker.log)"' 2>/dev/null
done
