#!/usr/bin/env bash
# Gated pipeline for the combined frightened-timer fix. Any failed gate stops everything before training.
set -uo pipefail
cd /workspace/pacworld
PY=.venv/bin/python; A=m1-3M-r148-ft-fright; B=m1-3M-ctx6s16-ft-fright
st() { echo "[3m $(date -u +%T)] $*"; }
fail() { st "GATE FAILED: $*"; exit 1; }
while tmux has-session -t recpel 2>/dev/null; do sleep 15; done
grep -q "^done:" logs/record_pellet.log || fail "recording did not finish"
st "recording: $(grep -E '^done:|^seeking:' logs/record_pellet.log | tr '\n' ' ')"
st "phases new:   $($PY tools/count_phases.py data/pellet/agent)"
st "phases 2m:    $($PY tools/count_phases.py data/2m/agent)"
st "phases orig:  $($PY tools/count_phases.py data/agent)"
dups=$(ls data/agent data/2m/agent data/pellet/agent | grep "^ep_" | sort | uniq -d | wc -l); st "episode seeds present in more than one folder: $dups"
[ "$dups" -eq 0 ] || fail "duplicate episode seeds"
[ -f configs/val_episodes_3m.json ] || $PY tools/freeze_val_split.py --config configs/$A.yaml --seed 0 || fail "freeze split"
st "split sha256: $(sha256sum configs/val_episodes_3m.json)"
$PY - <<'PY' || fail "split does not contain the frozen 2M split"
import json
a = json.load(open("configs/val_episodes_3m.json")); b = json.load(open("configs/val_episodes_2m.json"))
assert set(b["val_episode_seeds"]) <= set(a["val_episode_seeds"]) and a["from_val_base"] == sorted(b["val_episode_seeds"])
print("  3m split:", len(a["val_episode_seeds"]), "val episodes =", len(a["from_val_base"]), "frozen +", len(a["from_val_new_folder"]), "new")
PY
st "build cache + window checks (r148 layout)"
$PY -u dataset.py --config configs/$A.yaml --seed 0 --rebuild --workers 24 --check-windows 12 2>&1 | grep -vE "Warning|^\s*$|^  cached" | tee logs/cache_3m.log
grep -q "history buffer == dataset window" logs/cache_3m.log || fail "window checks (a)"
st "window checks (ctx6s16 layout)"
$PY -u dataset.py --config configs/$B.yaml --seed 0 --check-windows 12 2>&1 | grep -vE "Warning|^\s*$" | tee logs/cache_3m_b.log
grep -q "history buffer == dataset window" logs/cache_3m_b.log || fail "window checks (b)"
st "cache sha256 (meta): $(sha256sum data/cache/frames64_3m_meta.npz | cut -c1-16); size $(ls -la data/cache/frames64_3m.npy | awk '{print $5}')"
$PY tools/build_event_index.py --config configs/$A.yaml --seed 0 2>&1 | grep -vE "Warning|^\s*$" || fail "event index"
st "warming the page cache"; cat data/cache/frames64_3m.npy > /dev/null
$PY -m wandb login --verify >/dev/null 2>&1 || .venv/bin/wandb login --verify >/dev/null 2>&1 || fail "wandb login"
sha256sum checkpoints/m1-2M-ctx-r148/model1_latest.pt checkpoints/m1-2M-ctx-r148/model1_ema.pt checkpoints/m1-2M-ctx6s16-ft-uniform/model1_latest.pt checkpoints/m1-2M-ctx6s16-ft-uniform/model1_ema.pt > logs/parent_sha_before_3m.txt
for RUN in $A $B; do
  st "smoke $RUN"
  $PY -u train_model1.py --config configs/$RUN.yaml --seed 0 --steps 150 --eval-every 150 --wandb-mode disabled --run-name smoke 2>&1 | grep -E "step    150|eval @|done:|fine-tuning|event windows|episodes |Traceback|Error|Killed" | tee logs/smoke_$RUN.log
  grep -q "^done:" logs/smoke_$RUN.log || fail "smoke $RUN"
  rm -rf checkpoints/$RUN outputs/$RUN logs/$RUN.log logs/$RUN.done
done
st "GATES PASSED - launching training"
evalchain() {
  RUN=$1; PARENT_EVAL=$2; CFG=configs/eval_$RUN.yaml
  until [ -f logs/$RUN.done ]; do sleep 20; grep -q "$RUN exited with status [1-9]" logs/queue_2m.log && { st "FAILED training $RUN"; return 1; }; done
  st "$RUN trained: $(grep -E '^done:' logs/$RUN.log | tail -1)"
  $PY eval/eval_rollouts.py --config $CFG --seed 0 --model model1 --save-all-preds > logs/eval_$RUN.log 2>&1 && $PY eval/ghost_diagnostics.py --config $CFG --seed 0 >> logs/eval_$RUN.log 2>&1 && \
  $PY eval/pen_timer_analysis.py --config $CFG --seed 0 >> logs/eval_$RUN.log 2>&1 && $PY eval/residual_ghosts.py --config $CFG --seed 0 >> logs/eval_$RUN.log 2>&1 && \
  $PY eval/fair_ghosts.py --config $CFG --seed 0 >> logs/eval_$RUN.log 2>&1 && $PY eval/tf_phase_end.py $PARENT_EVAL $CFG >> logs/eval_$RUN.log 2>&1 && st "$RUN EVAL DONE" || st "FAILED eval $RUN"
}
bash tools/run_2m_queue.sh $A $B > logs/queue_3m_stdout.log 2>&1 &
evalchain $A configs/eval_m1-2M-ctx-r148.yaml &
evalchain $B configs/eval_m1-2M-ctx6s16-ft-uniform.yaml &
wait
st "parents untouched: $(sha256sum -c logs/parent_sha_before_3m.txt 2>&1 | tr '\n' ' ')"
$PY eval/compare_runs.py --seed 0 --out eval/results/compare_3m.csv --runs m1-2M-ctx6s16-ft-uniform=eval/results/m1-2M-ctx6s16-ft-uniform m1-2M-ctx-r148=eval/results/m1-2M-ctx-r148 $A=eval/results/$A $B=eval/results/$B > logs/compare_3m.log 2>&1
st "ALL DONE"
