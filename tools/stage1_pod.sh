#!/usr/bin/env bash
# Stage 1 of the latent-action project on the pod (pre-registration 6bb6039; design notes/latent_actions_design.md).
#   bash tools/stage1_pod.sh phase1        firewall double runs on the real configs, LAM arms A and B (30k steps each),
#                                          codes, label-free checks L1-L3 + T_det, the arm choice. NO labels are read.
#                                          -> the choice is committed to notes/handoff.md from the Mac before phase2.
#   bash tools/stage1_pod.sh phase2 <A|B>  prior + key map for the chosen arm, then eval/lam_agreement.py (G1, G2).
# wandb: this pod is not logged in, so training runs are offline and synced from the Mac (wandb/offline-run-*).
# Every step logs to logs/stage1_*.log; a failed step stops the script.
set -uo pipefail
cd /workspace/pacworld
PY=.venv/bin/python
WB="--wandb-mode ${WANDB_MODE_OVERRIDE:-offline}"
st() { echo "[stage1 $(date -u +%T)] $*"; }
fail() { st "FAILED: $*"; exit 1; }

phase1() {
  [ -f data/cache/pac64_2m.npy ] || fail "data/cache/pac64_2m.npy missing (tools/pac_positions.py)"
  grep -q "OK (>= 0.76)" logs/stage0_ceiling_gate.log 2>/dev/null || fail "full-split ceiling precondition not recorded as OK"
  st "firewall double runs on the real configs (150 steps, wandb disabled)"
  for ARM in A B; do
    $PY -u tools/firewall_test.py --seed 0 --report logs/firewall_pod.jsonl -- $PY -u train_lam.py --config configs/lam-$ARM.yaml \
      --seed 0 --part lam --steps 150 --eval-every 150 --wandb-mode disabled > logs/stage1_firewall_$ARM.log 2>&1 \
      || fail "firewall test lam-$ARM (logs/stage1_firewall_$ARM.log)"
    grep "FIREWALL TEST PASSED" logs/stage1_firewall_$ARM.log || fail "firewall lam-$ARM did not pass"
    grep -E "it/s" logs/stage1_firewall_$ARM.log | tail -1
  done
  rm -rf checkpoints/lam-A checkpoints/lam-B outputs/lam-A outputs/lam-B
  for ARM in A B; do
    st "train lam-$ARM (30k steps)"
    $PY -u train_lam.py --config configs/lam-$ARM.yaml --seed 0 --part lam --run-name lam-$ARM $WB > logs/stage1_train_$ARM.log 2>&1 \
      || fail "training lam-$ARM (logs/stage1_train_$ARM.log)"
    grep -E "eval @ 30000|done:" logs/stage1_train_$ARM.log
    st "codes lam-$ARM"
    $PY -u tools/lam_codes.py --config configs/lam-$ARM.yaml --seed 0 > logs/stage1_codes_$ARM.log 2>&1 || fail "codes lam-$ARM"
    tail -n 1 logs/stage1_codes_$ARM.log
    st "label-free checks lam-$ARM"
    $PY -u tools/lam_checks.py --config configs/lam-$ARM.yaml --seed 0 > logs/stage1_checks_$ARM.log 2>&1 || fail "checks lam-$ARM"
    tail -n 1 logs/stage1_checks_$ARM.log
  done
  st "arm choice (label-free)"
  $PY -u tools/lam_checks.py --select configs/lam-A.yaml configs/lam-B.yaml --seed 0 | tee logs/stage1_select.log
  st "phase1 done: commit the choice (checkpoints/lam_selection.json) before phase2"
}

phase2() {
  ARM=$1
  grep -q "\"choice\": \"lam-$ARM\"" checkpoints/lam_selection.json || fail "lam-$ARM is not the recorded choice"
  st "prior lam-$ARM"
  $PY -u train_lam.py --config configs/lam-$ARM.yaml --seed 0 --part prior --run-name lam-$ARM-prior $WB > logs/stage1_prior_$ARM.log 2>&1 \
    || fail "prior lam-$ARM"
  grep -E "eval @|done:" logs/stage1_prior_$ARM.log | tail -2
  st "key map and resolver thresholds lam-$ARM"
  $PY -u tools/lam_keymap.py --config configs/lam-$ARM.yaml --seed 0 > logs/stage1_keymap_$ARM.log 2>&1 || fail "keymap lam-$ARM"
  tail -n 1 logs/stage1_keymap_$ARM.log
  st "G1 / G2 (labels, evaluation only) lam-$ARM"
  $PY -u eval/lam_agreement.py --config configs/lam_agreement.yaml --lam-config configs/lam-$ARM.yaml --seed 0 \
    > logs/stage1_agreement_$ARM.log 2>&1 || fail "lam_agreement lam-$ARM"
  tail -n 1 logs/stage1_agreement_$ARM.log
}

case "${1:-}" in
  phase1) phase1 ;;
  phase2) phase2 "${2:?arm A or B}" ;;
  *) echo "usage: $0 phase1 | phase2 <A|B>"; exit 2 ;;
esac
