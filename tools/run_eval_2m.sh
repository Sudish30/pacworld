#!/usr/bin/env bash
# Full evaluation of the three 2M-dataset checkpoints plus the comparison against Model 1. Run on the pod:
#   tmux new -d -s eval2m "bash tools/run_eval_2m.sh 2>&1 | tee logs/eval_2m.log"
# Per run (~25 min each on a 4090): rollout eval with all frames saved -> ghost diagnostics -> pen-timer analysis.
# Model 1 only needs pen_timer_analysis.py re-run (its rollouts and diagnostics exist) to get pen_metrics.csv.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
for run in m1-2M-ctx4 m1-2M-ctx8 m1-2M-ctx6s16; do
  cfg=configs/eval_$run.yaml
  echo "[eval $(date -u +%FT%TZ)] $run: rollouts"
  $PY eval/eval_rollouts.py --config $cfg --seed 0 --model model1 --save-all-preds || { echo "[eval] $run rollouts FAILED"; continue; }
  echo "[eval $(date -u +%FT%TZ)] $run: ghost diagnostics"
  $PY eval/ghost_diagnostics.py --config $cfg --seed 0 || echo "[eval] $run ghost_diagnostics FAILED"
  echo "[eval $(date -u +%FT%TZ)] $run: pen timer"
  $PY eval/pen_timer_analysis.py --config $cfg --seed 0 || echo "[eval] $run pen_timer FAILED"
done
echo "[eval $(date -u +%FT%TZ)] model1: pen timer (pen_metrics.csv)"
$PY eval/pen_timer_analysis.py --config configs/eval.yaml --seed 0
$PY eval/compare_runs.py --seed 0 --runs model1=eval/results m1-2M-ctx4=eval/results/m1-2M-ctx4 m1-2M-ctx8=eval/results/m1-2M-ctx8 m1-2M-ctx6s16=eval/results/m1-2M-ctx6s16
$PY eval/eval_rollouts.py --seed 0 --plot eval/results/model1 eval/results/m1-2M-ctx4/model1 eval/results/m1-2M-ctx8/model1 eval/results/m1-2M-ctx6s16/model1
echo "[eval $(date -u +%FT%TZ)] all done"
