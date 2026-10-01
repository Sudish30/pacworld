#!/usr/bin/env bash
# Training-seed replication of the hidden-timer follow-up (notes/timer_rule_design.md section 12) on a GPU pod.
# Stage 1 cells first, then stage 2. Resumable: a scored cell is skipped. Logs: logs/timer/<layout>-N<N>-s<S>.log
#   cd /workspace/pacworld && JOBS=3 nohup bash tools/timer_seeds.sh > logs/timer_seeds.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
JOBS=${JOBS:-3}
mkdir -p logs/timer
jobs=$($PY -c "
import yaml
r = yaml.safe_load(open('configs/timer.yaml'))['seed_replication']
order = r['stage1'] + [c for c in r['cells'] if c not in r['stage1']]
print(' '.join(f'{c}:{s}' for c in order for s in r['cells'][c]))")
run_cell() {
  IFS=: read -r lay N S <<< "$1"
  local name=$lay-N$N-s$S log=logs/timer/$lay-N$N-s$S.log
  [ -f "eval/results/timer/$name.json" ] && { echo "skip $name (scored)"; return 0; }
  if ! $PY -c "import sys,torch,yaml,pathlib; c=yaml.safe_load(open('configs/timer.yaml')); ck=pathlib.Path(c['train']['checkpoint_root'])/'$name'/'model.pt'; sys.exit(0 if ck.exists() and torch.load(ck,map_location='cpu')['step']==c['train']['steps'] else 1)" 2>/dev/null
  then
    $PY -u train_timer.py --layout "$lay" --N "$N" --seed "$S" --wandb-mode "${WANDB_MODE_OVERRIDE:-offline}" > "$log" 2>&1 || { echo "TRAIN FAILED $name (see $log)"; return 1; }
  fi
  $PY -u eval/timer_eval.py --layout "$lay" --N "$N" --seed 0 --train-seed "$S" >> "$log" 2>&1 || { echo "EVAL FAILED $name"; return 1; }
  tail -1 "$log"
}
export -f run_cell; export PY
echo $jobs | tr ' ' '\n' | xargs -P "$JOBS" -I{} bash -c 'run_cell {}'
echo "timer seed replication finished; score with: $PY tools/timer_seeds_verdict.py"
