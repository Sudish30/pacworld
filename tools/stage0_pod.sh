#!/usr/bin/env bash
# Stage 0 of the latent-action project on the pod (notes/handoff.md, pre-registration 6bb6039). Measures and reports;
# it deletes NOTHING (the 128px cache is removed by hand only after its users and sha256 are recorded) and trains
# nothing beyond 150-step smoke runs. Every step logs to logs/stage0_*.log.
#   cd /workspace/pacworld && bash tools/stage0_pod.sh
set -uo pipefail
cd /workspace/pacworld
PY=.venv/bin/python
st() { echo "[stage0 $(date -u +%T)] $*"; }
fail() { st "FAILED: $*"; exit 1; }

st "1. disk"
{ df -h /workspace; du -sh data/cache/* checkpoints eval/results data/*/ 2>/dev/null; } | tee logs/stage0_disk.log

st "2. the 128px files: sha256 and users (nothing is deleted here)"
{ ls -l data/cache/frames128_2m.npy data/cache/frames128_2m_meta.npz
  sha256sum data/cache/frames128_2m.npy data/cache/frames128_2m_meta.npz
  echo "--- references in code/configs:"; grep -rln frames128 configs tools eval serve ./*.py 2>/dev/null
  echo "--- running python processes:"; ps aux | grep -E "[p]ython" || echo "none"
  echo "--- open handles:"; (command -v lsof >/dev/null && lsof data/cache/frames128_2m.npy) || echo "none (or lsof missing)"
} 2>&1 | tee logs/stage0_sha128.log

st "3. frozen split and wandb"
[ "$(sha256sum configs/val_episodes_2m.json | cut -c1-16)" = "7ba75f8952e767b3" ] || fail "val split is not the frozen one"
if .venv/bin/wandb login --verify > /dev/null 2>&1; then st "   split sha256 ok, wandb logged in"
else st "   split sha256 ok; wandb NOT logged in on this pod: runs use --wandb-mode offline and are synced from the Mac"; fi

st "4. Pac-Man positions over frames64_2m.npy (pixel detector)"
[ -f data/cache/pac64_2m.npy ] && st "   exists, skipped" || \
  $PY -u tools/pac_positions.py --config configs/lam-A.yaml --seed 0 --workers "$(nproc)" 2>&1 | tail -3 | tee logs/stage0_pac.log

st "5. visible-action ceiling on the full 388-episode val split (precondition: lag-1 pixel ceiling >= 0.76)"
$PY -u eval/visible_ceiling.py --config configs/lam_agreement.yaml --seed 0 --out eval/results/lam/visible_ceiling_full.json 2>&1 \
  | tee logs/stage0_ceiling.log | grep -E "episodes|decision_events|pix_lag1 -> new|ram1 -> new|majority|new_direction\""
$PY - <<'EOF' | tee -a logs/stage0_ceiling.log
import json
c = json.load(open("eval/results/lam/visible_ceiling_full.json"))
v = c["ceiling"]["pix_lag1 -> new_direction"]
print(f"PRECONDITION lag-1 pixel ceiling {v:.4f} on {c['decision_events']} events of {c['episodes']} episodes: "
      + ("OK (>= 0.76)" if v >= 0.76 else "BELOW 0.76 -> STOP AND ASK THE OWNER"))
EOF

st "6. smoke runs (150 steps, wandb disabled) and the pod firewall test on the real configs"
for ARM in A B; do
  $PY -u tools/firewall_test.py --seed 0 --report logs/firewall_pod.jsonl -- $PY -u train_lam.py --config configs/lam-$ARM.yaml \
    --seed 0 --part lam --steps 150 --eval-every 150 --wandb-mode disabled 2>&1 | grep -E "it/s|eval @|FIREWALL|params|Traceback|Error" \
    | tail -5 | tee -a logs/stage0_smoke.log
done
rm -rf checkpoints/lam-A checkpoints/lam-B outputs/lam-A outputs/lam-B     # smoke weights only; Stage 1 starts clean
st "done. Read logs/stage0_*.log. The 128px deletion is a separate, manual step."
