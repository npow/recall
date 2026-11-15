#!/bin/bash
# Score Lambda dev episodes as they are synced back (3 independent Gemini calls each). Stop: touch outputs/director-watch.stop
cd "$(dirname "$0")/.."
while [ ! -f outputs/director-watch.stop ]; do
  /home/npow/code/iteratehack/.venv/bin/python scripts/director_score.py lambda --workers 10 >> outputs/score-lambda.log 2>&1
  sleep 45
done
