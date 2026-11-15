#!/usr/bin/env bash
# Bring one instance into the final run (run LOCALLY, one invocation per instance):
#   scripts/lambda_bringup_final.sh IP SOURCE_WORKTREE REMOTE_NAME JOBS_FOR_THIS_IP LABEL [skip-setup]
# Waits for SSH, builds the base environment with verified weights (unless skip-setup), pushes the worktree as
# ~/REMOTE_NAME, opens the reverse tunnel to the local player proxy, and starts a plain worker with 2 retries.
set -euo pipefail
IP="$1"; SOURCE="$2"; REMOTE="$3"; JOBS="$4"; LABEL="$5"; SKIP="${6:-}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
until ssh -i "${LAMBDA_SSH_KEY:-$HOME/.ssh/id_rsa}" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "ubuntu@$IP" true 2>/dev/null; do sleep 15; done
echo "$(date +%T) ssh up $IP"
if [ "$SKIP" != "skip-setup" ]; then
  python3 scripts/lambda_fleet.py setup "$IP"
  echo "$(date +%T) base ready $IP"
fi
python3 scripts/lambda_fleet.py --remote "$REMOTE" push --source "$SOURCE" "$IP"
if ! pgrep -f "8317:127.0.0.1:8317 ubuntu@$IP\$" >/dev/null; then python3 scripts/lambda_fleet.py tunnel "$IP"; sleep 4; fi
python3 scripts/lambda_fleet.py --remote "$REMOTE" start --jobs "$JOBS" --plain "$LABEL" --plain-retries 2 --shards 1 "$IP"
echo "$(date +%T) worker started $IP"
