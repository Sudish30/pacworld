#!/usr/bin/env bash
# Gated pipeline for the 128x128 polish run. Every gate that fails stops everything before GPU time is spent.
set -uo pipefail
cd /workspace/pacworld
PY=.venv/bin/python; A=m1-2M-128-ctx6s16; B=m1-2M-128-ctx6s16-ft-uniform
st() { echo "[128 $(date -u +%T)] $*"; }
fail() { st "GATE FAILED: $*"; exit 1; }
while tmux has-session -t b128 2>/dev/null; do sleep 20; done
grep -q "ROUNDTRIP OK" logs/build128.log || fail "palette round trip"
grep -q "history buffer == dataset window" logs/build128.log || fail "window checks"
grep -qE "every one of the .* pixels mapped exactly" logs/build128.log || fail "full-cache palette mapping"
st "cache: $(grep -E '^wrote data/cache/frames128|round trip:|^size:' logs/build128.log | tr '\n' ' ')"
st "split sha256 $(sha256sum configs/val_episodes_2m.json | cut -c1-16) (frozen: 7ba75f8952e767b3)"
[ "$(sha256sum configs/val_episodes_2m.json | cut -c1-16)" = "7ba75f8952e767b3" ] || fail "split"
.venv/bin/wandb login --verify >/dev/null 2>&1 || fail "wandb login"
st "warming the page cache"; cat data/cache/frames128_2m.npy > /dev/null
st "smoke $A"
$PY -u train_model1.py --config configs/$A.yaml --seed 0 --steps 150 --eval-every 150 --wandb-mode disabled --run-name smoke 2>&1 \
  | grep -E "step    150|eval @|done:|params|palette cache|episodes |Traceback|Error|Killed" | tee logs/smoke_$A.log
grep -q "^done:" logs/smoke_$A.log || fail "smoke $A"
rm -rf checkpoints/$A outputs/$A logs/$A.log logs/$A.done
st "GATES PASSED - training $A (100k)"
bash tools/run_2m_queue.sh $A > logs/queue_128a.log 2>&1
[ -f logs/$A.done ] || fail "training $A"
st "$A trained: $(grep -E '^done:' logs/$A.log | tail -1)"
sha256sum checkpoints/$A/model1_latest.pt checkpoints/$A/model1_ema.pt > logs/parent_sha_128.txt
st "smoke $B"
$PY -u train_model1.py --config configs/$B.yaml --seed 0 --steps 150 --eval-every 150 --wandb-mode disabled --run-name smoke 2>&1 \
  | grep -E "step    150|eval @|done:|fine-tuning|Traceback|Error|Killed" | tee logs/smoke_$B.log
grep -q "^done:" logs/smoke_$B.log || fail "smoke $B"
rm -rf checkpoints/$B outputs/$B logs/$B.log logs/$B.done
st "training $B (15k anneal)"
bash tools/run_2m_queue.sh $B > logs/queue_128b.log 2>&1
[ -f logs/$B.done ] || fail "training $B"
st "$B trained: $(grep -E '^done:' logs/$B.log | tail -1); parent untouched: $(sha256sum -c logs/parent_sha_128.txt 2>&1 | tr '\n' ' ')"
CFG=configs/eval_$B.yaml
for step in "eval_rollouts.py --config $CFG --seed 0 --model model1 --save-all-preds" "ghost_diagnostics.py --config $CFG --seed 0" \
            "pen_timer_analysis.py --config $CFG --seed 0" "residual_ghosts.py --config $CFG --seed 0" "fair_ghosts.py --config $CFG --seed 0"; do
  st "eval: ${step%% *}"
  $PY eval/$step >> logs/eval_$B.log 2>&1 || fail "eval ${step%% *}"
done
$PY eval/compare_runs.py --seed 0 --out eval/results/compare_128.csv --runs m1-2M-ctx6s16-ft-uniform=eval/results/m1-2M-ctx6s16-ft-uniform $B=eval/results/$B > logs/compare_128.log 2>&1
st "demo 15 fps test (temporary instance on :8765; the production demo is not touched)"
sed -e "s#^checkpoint: .*#checkpoint: checkpoints/$B/model1_ema.pt#" -e "s/^port: .*/port: 8765/" configs/serve.yaml > /tmp/serve128.yaml
tmux kill-session -t s128 2>/dev/null; tmux new -d -s s128 "$PY -u serve/server.py --seed 0 --config /tmp/serve128.yaml > logs/serve128_test.log 2>&1"
for i in $(seq 1 90); do sleep 2; curl -sf localhost:8765/status >/dev/null && break; done
$PY /tmp/wd_test.py 8765 460 | tee logs/demo128_fps.log
curl -s localhost:8765/status | tee -a logs/demo128_fps.log; echo
tmux kill-session -t s128
st "ALL DONE"
