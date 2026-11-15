#!/usr/bin/env bash
# Bring one freshly launched instance into a running dev batch (run LOCALLY, one invocation per instance):
#   scripts/lambda_bringup.sh IP SHARD SHARDS SOURCE_WORKTREE REMOTE_NAME JOBS_JSONL LABEL
# Waits for SSH, builds the base environment (weights verified), pushes the given worktree as ~/REMOTE_NAME and
# starts a fixed-action (no player) plain worker on shard SHARD of SHARDS.
set -euo pipefail
IP="$1"; SHARD="$2"; SHARDS="$3"; SOURCE="$4"; REMOTE="$5"; JOBS="$6"; LABEL="$7"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
until ssh -i "${LAMBDA_SSH_KEY:-$HOME/.ssh/id_rsa}" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "ubuntu@$IP" true 2>/dev/null; do sleep 15; done
echo "$(date +%T) ssh up $IP"
python3 scripts/lambda_fleet.py setup "$IP"
echo "$(date +%T) base ready $IP"
python3 scripts/lambda_fleet.py --remote "$REMOTE" push --source "$SOURCE" "$IP"
python3 scripts/lambda_fleet.py --remote "$REMOTE" start --jobs "$JOBS" --plain "$LABEL" --no-player --shards "$SHARDS" --first-shard "$SHARD" "$IP"
echo "$(date +%T) worker started $IP shard $SHARD/$SHARDS"
