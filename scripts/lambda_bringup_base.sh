#!/usr/bin/env bash
# Prepare one freshly launched instance as a standby (run LOCALLY): wait for SSH, build the base environment with
# verified weights, open the reverse tunnel to the local player proxy. No job is started.
#   scripts/lambda_bringup_base.sh IP
set -euo pipefail
IP="$1"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
until ssh -i "${LAMBDA_SSH_KEY:-$HOME/.ssh/id_rsa}" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "ubuntu@$IP" true 2>/dev/null; do sleep 15; done
echo "$(date +%T) ssh up $IP"
python3 scripts/lambda_fleet.py setup "$IP"
python3 scripts/lambda_fleet.py tunnel "$IP"
echo "$(date +%T) standby ready $IP"
