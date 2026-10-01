#!/usr/bin/env bash
# Part b, attempt 2 (notes/timer_rule_design.md section 11), on a GPU pod. pac-S10 first; P-S is the gate: pac-C10 is
# trained only if pac-S10-ft-uniform passes P-S. Resumable: finished stages are skipped.
#   cd /workspace/pacworld && nohup bash tools/partb_attempt2.sh > logs/partb_attempt2.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
WB=${WANDB_MODE_OVERRIDE:-offline}
R=eval/results/pacman
step_of() { $PY -c "import sys,torch; print(torch.load(sys.argv[1], map_location='cpu')['step'])" "$1" 2>/dev/null || echo 0; }

train() {   # train <layout>: 100k steps (resuming if a checkpoint exists), then the 15k-step anneal
  local run=pac-$1 latest=checkpoints/pac-$1/model1_latest.pt
  if [ "$(step_of $latest)" != 100000 ]; then
    if [ -f $latest ]; then
      $PY -u train_model1.py --config configs/$run.yaml --seed 0 --resume $latest --wandb-mode $WB --run-name $run-attempt2 || exit 1
    else
      $PY -u train_model1.py --config configs/$run.yaml --seed 0 --wandb-mode $WB --run-name $run-attempt2 || exit 1
    fi
  fi
  if [ "$(step_of checkpoints/$run-ft-uniform/model1_latest.pt)" != 15000 ]; then
    $PY -u train_model1.py --config configs/$run-ft-uniform.yaml --seed 0 --wandb-mode $WB --run-name $run-ft-uniform || exit 1
  fi
}

[ -f $R/S10-pilot.json ] && [ ! -f $R/S10-pilot20k.json ] && mv $R/S10-pilot.json $R/S10-pilot20k.json   # attempt 1's stopped pilot
train S10
if [ ! -f $R/S10-100k.json ]; then      # reported, not gating: before the anneal
  $PY -u eval/pacman_house.py --layout S10 --seed 0 --checkpoint checkpoints/pac-S10/model1_ema.pt && mv $R/S10-pilot.json $R/S10-100k.json
fi
[ -f $R/S10.json ] || $PY -u eval/pacman_house.py --layout S10 --seed 0 || exit 1
if $PY -c "
import json, sys
r = json.load(open('$R/S10.json')); m, i = r['model']['point'], r['ideal']['point']
rb, mb = min(0.90, i['release_frac'] - 0.10), i['median_abs_err'] + 3
ok = m['release_frac'] >= rb and m['median_abs_err'] <= mb
print(f\"P-S: release {m['release_frac']:.3f} (bar {rb:.3f}), median error {m['median_abs_err']:.1f} (bar {mb:.1f}) -> {'PASS' if ok else 'FAIL'}\")
sys.exit(0 if ok else 1)"
then
  train C10
  [ -f $R/C10.json ] || $PY -u eval/pacman_house.py --layout C10 --seed 0 || exit 1
  echo "PART B ATTEMPT 2 FINISHED: S10 and C10 scored"
else
  echo "PART B ATTEMPT 2 STOPPED: P-S failed; pac-C10 is not trained"
fi
