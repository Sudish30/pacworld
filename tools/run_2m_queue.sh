#!/usr/bin/env bash
# The three 100k-step runs on the 2M-step dataset, one after the other. Start inside tmux:
#   tmux new -d -s train2m "bash tools/run_2m_queue.sh"
# Each run reads configs/<name>.yaml, logs to wandb as <name> and to logs/<name>.log, and keeps its
# checkpoints in checkpoints/<name>/. Safe to start again after an interruption: finished runs are
# skipped and an unfinished run resumes from its last checkpoint. A failed run does not stop the queue.
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
SEED=0   # same seed as Model 1
for name in m1-2M-ctx4 m1-2M-ctx8 m1-2M-ctx6s16; do
  if [ -f "logs/$name.done" ]; then echo "[queue] $name already finished, skipping"; continue; fi
  resume=()
  if [ -f "checkpoints/$name/model1_latest.pt" ]; then resume=(--resume "checkpoints/$name/model1_latest.pt"); fi
  echo "[queue] $(date -u +%FT%TZ) starting $name ${resume[*]}" | tee -a logs/queue_2m.log
  .venv/bin/python -u train_model1.py --config "configs/$name.yaml" --seed $SEED --run-name "$name" "${resume[@]}" 2>&1 | tee -a "logs/$name.log"
  status=${PIPESTATUS[0]}
  echo "[queue] $(date -u +%FT%TZ) $name exited with status $status" | tee -a logs/queue_2m.log
  if [ "$status" -eq 0 ]; then touch "logs/$name.done"; fi
done
echo "[queue] $(date -u +%FT%TZ) all runs done" | tee -a logs/queue_2m.log
