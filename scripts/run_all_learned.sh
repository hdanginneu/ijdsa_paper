#!/bin/bash
# Run every learned model after its tuning file exists (tuning runs separately).
cd "$(dirname "$0")/.."
for job in "svr full" "rf full" "xgb full" "xgb price" "dnn full" "gru full" "gru price" "lstm full"; do
  set -- $job
  while [ ! -f results/tuning/$1_$2.json ]; do sleep 30; done
  python3 scripts/run_learned.py $1 $2
done
