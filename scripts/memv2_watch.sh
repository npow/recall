#!/bin/bash
# Score memory-v2 step-2 episodes as they land: one Gemini call each first, then two more.
# Stock judgments are reused from the final1 run (same judge and protocol, three calls per video).
cd "$(dirname "$0")/.."
PY=/home/npow/code/iteratehack/.venv-lingbot/bin/python
while true; do
  $PY scripts/memv2_score.py --memv2 outputs/memv2-step2 --out outputs/memv2-step2-judge --reps 1 --workers 8 >> outputs/memv2-step2-score.log 2>&1
  $PY scripts/memv2_score.py --memv2 outputs/memv2-step2 --out outputs/memv2-step2-judge --reps 3 --workers 8 >> outputs/memv2-step2-score.log 2>&1
  [ -f outputs/memv2-step2-stop ] && break
  sleep 60
done
