#!/usr/bin/env bash
# Train and score every pre-registered cell of the hidden-timer experiment (notes/timer_rule_design.md) on a GPU pod.
# Resumable: a cell with a finished checkpoint (step == train.steps) is not retrained; a cell with a result JSON is
# not re-scored. Runs JOBS cells at a time (the models are tiny). Logs: logs/timer/<layout>-N<N>.log
#   cd /workspace/pacworld && JOBS=3 bash tools/timer_sweep.sh
#   GRID=followup_grid JOBS=2 bash tools/timer_sweep.sh     # the follow-up cells (design section 10)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
JOBS=${JOBS:-3}
mkdir -p logs/timer
cells=$($PY - <<'EOF'
import yaml
c = yaml.safe_load(open("configs/timer.yaml"))
import os
print(" ".join(f"{l}:{n}" for l, ns in c[os.environ.get("GRID", "grid")].items() for n in ns))
EOF
)
run_cell() {
  local lay=${1%%:*} N=${1##*:} log=logs/timer/${1%%:*}-N${1##*:}.log
  [ -f "eval/results/timer/$lay-N$N.json" ] && { echo "skip $lay N=$N (scored)"; return 0; }
  # a cell with a finished checkpoint (step == train.steps) is not retrained (plain -c: a heredoc does not survive export -f)
  if ! $PY -c "import sys,torch,yaml,pathlib; c=yaml.safe_load(open('configs/timer.yaml')); ck=pathlib.Path(c['train']['checkpoint_root'])/'$lay-N$N'/'model.pt'; sys.exit(0 if ck.exists() and torch.load(ck,map_location='cpu')['step']==c['train']['steps'] else 1)" 2>/dev/null
  then
    $PY -u train_timer.py --layout "$lay" --N "$N" --seed 0 --wandb-mode "${WANDB_MODE_OVERRIDE:-offline}" > "$log" 2>&1 || { echo "TRAIN FAILED $lay N=$N (see $log)"; return 1; }
  fi
  $PY -u eval/timer_eval.py --layout "$lay" --N "$N" --seed 0 >> "$log" 2>&1 || { echo "EVAL FAILED $lay N=$N"; return 1; }
  tail -1 "$log"
}
export -f run_cell; export PY
echo $cells | tr ' ' '\n' | xargs -P "$JOBS" -I{} bash -c 'run_cell {}'
echo "timer sweep finished; score with: $PY tools/timer_verdict.py"
