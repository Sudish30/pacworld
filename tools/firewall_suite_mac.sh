#!/usr/bin/env bash
# Label-firewall suite on the Mac smoke configs (tools/firewall_test.py for each training entry point).
# Every run that must not see labels has to PASS (bit-identical with the recorded actions scrambled); the labeled
# negative control m1-labels-mac has to FAIL, which shows the test can detect a run that reads labels.
# Needs data/cache/frames64.npy, pac64.npy and lam_codes_B_mac.npy (tools/pac_positions.py, tools/lam_codes.py).
#   bash tools/firewall_suite_mac.sh [seed]      -> logs/firewall_mac.jsonl, exit 0 only if every expectation holds
set -u
cd "$(dirname "$0")/.."
SEED=${1:-0}
REPORT=logs/firewall_mac.jsonl
mkdir -p logs && : > "$REPORT"
PY=.venv/bin/python
LAM="--steps 20 --batch-size 8 --eval-every 10 --wandb-mode disabled"
WM="--steps 10 --eval-every 10 --wandb-mode disabled"
status=0
expect() {  # expect <pass|fail> <command...>
  local want=$1; shift
  $PY tools/firewall_test.py --seed "$SEED" --report "$REPORT" -- "$@" > "logs/firewall_last.log" 2>&1
  local rc=$?
  local got=$([ $rc -eq 0 ] && echo pass || echo fail)
  grep -q '"pass"' logs/firewall_last.log || got="error"
  printf '%-6s expected %-4s: %s\n' "$got" "$want" "$*"
  [ "$got" = "$want" ] || status=1
}
expect pass $PY train_lam.py --config configs/lam-A-mac.yaml --seed "$SEED" --part lam $LAM
expect pass $PY train_lam.py --config configs/lam-B-mac.yaml --seed "$SEED" --part lam $LAM
expect pass $PY train_lam.py --config configs/lam-B-mac.yaml --seed "$SEED" --part prior $LAM
expect pass $PY train_model1.py --config configs/m1-noact-mac.yaml --seed "$SEED" $WM
expect pass $PY train_model1.py --config configs/m1-latent-mac.yaml --seed "$SEED" $WM
expect fail $PY train_model1.py --config configs/m1-labels-mac.yaml --seed "$SEED" $WM
echo "firewall suite: $([ $status -eq 0 ] && echo 'ALL EXPECTATIONS MET' || echo 'EXPECTATION VIOLATED')  ($REPORT)"
exit $status
