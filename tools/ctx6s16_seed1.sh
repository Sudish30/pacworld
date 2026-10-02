#!/usr/bin/env bash
# Training-seed replication of the strided Ms. Pac-Man model (notes/handoff.md, pre-registered 2026-10-02), on a GPU pod.
# Gates first (split checksum, CUDA, the old runs' saved rollouts, 150-step smoke run), then 100k steps with --seed 1, then the standard
# rollout evaluation and the pen-timer bootstrap. Resumable.
#   cd /workspace/pacworld && nohup bash tools/ctx6s16_seed1.sh > logs/ctx6s16_seed1.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUN=m1-2M-ctx6s16-s1
CFG=configs/$RUN.yaml
step_of() { $PY -c "import sys,torch; print(torch.load(sys.argv[1], map_location='cpu')['step'])" "$1" 2>/dev/null || echo 0; }
sha=$(sha256sum configs/val_episodes_2m.json | cut -c1-8)
[ "$sha" = "7ba75f89" ] || { echo "GATE FAILED: val split sha256 $sha != 7ba75f89"; exit 1; }
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "GATE FAILED: no CUDA"; exit 1; }
for r in m1-2M-ctx4 m1-2M-ctx6s16; do for f in preds_all.npy jobs.csv; do [ -f eval/results/$r/model1/$f ] || { echo "GATE FAILED: eval/results/$r/model1/$f missing"; exit 1; }; done; done
latest=checkpoints/$RUN/model1_latest.pt
if [ "$(step_of $latest)" != 100000 ]; then
  if [ -f $latest ]; then
    $PY -u train_model1.py --config $CFG --seed 1 --resume $latest --wandb-mode "${WANDB_MODE_OVERRIDE:-offline}" --run-name $RUN || exit 1
  else
    $PY -u train_model1.py --config configs/m1-2M-ctx6s16-s1-smoke.yaml --seed 1 --steps 150 --wandb-mode disabled || { echo "GATE FAILED: smoke run"; exit 1; }
    rm -rf checkpoints/$RUN-smoke outputs/$RUN-smoke
    $PY -u train_model1.py --config $CFG --seed 1 --wandb-mode "${WANDB_MODE_OVERRIDE:-offline}" --run-name $RUN || exit 1
  fi
fi
E=configs/eval_$RUN.yaml
[ -f eval/results/$RUN/model1/summary.csv ] || $PY eval/eval_rollouts.py --config $E --seed 0 --model model1 --save-all-preds || exit 1
$PY eval/ghost_diagnostics.py --config $E --seed 0 || echo "ghost_diagnostics FAILED (not part of the registered criterion)"
$PY eval/pen_timer_analysis.py --config $E --seed 0 || echo "pen_timer_analysis FAILED (not part of the registered criterion)"
$PY eval/pen_bootstrap.py --config configs/pen_stats_s1.yaml --seed 0 || exit 1
$PY -c "import json,sys; r=json.load(open('eval/results/stats_s1/pen_bootstrap.json'))['450']; sys.exit(0 if 'ctx6s16-s1 - ctx4' in r['pairs'] and r['runs']['ctx6s16-s1']['rollouts']==30 and all(r['runs'][n].get('reproduces_original') for n in ('ctx4','ctx6s16')) else 1)" || { echo "PIPELINE CHECK FAILED"; exit 1; }
echo "CTX6S16 SEED 1 FINISHED"
