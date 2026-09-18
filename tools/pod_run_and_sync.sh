#!/usr/bin/env bash
# Train one run on an extra pod, sync the results back to pod 1, and stop this pod.
#
#   bash tools/pod_run_and_sync.sh <run-name> <pod1-host> <pod1-ssh-port>
#
# After train_model1.py exits with status 0: rsync checkpoints/<run>/, outputs/<run>/ and
# logs/<run>.log to the same paths on pod 1, verify the EMA checkpoint's checksum on pod 1,
# then stop this pod with runpodctl if RUNPOD_API_KEY is set (GPU billing stops; the volume
# stays until the pod is removed). Everything is appended to logs/sync_<run>.log.
# A failed run or failed sync leaves the pod running so the state can be inspected.
set -uo pipefail
RUN=$1; P1HOST=$2; P1PORT=$3
cd "$(dirname "$0")/.." || exit 1
# RunPod exposes the pod's env (RUNPOD_POD_ID, RUNPOD_API_KEY, ...) to shells through this file;
# a tmux command is a non-interactive shell and does not get it from .bashrc.
[ -f /etc/rp_environment ] && source /etc/rp_environment
P1="ssh -p $P1PORT -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
log() { echo "[sync $(date -u +%FT%TZ)] $*" | tee -a "logs/sync_$RUN.log"; }
mkdir -p logs

resume=()
[ -f "checkpoints/$RUN/model1_latest.pt" ] && resume=(--resume "checkpoints/$RUN/model1_latest.pt")
log "starting $RUN ${resume[*]} on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
.venv/bin/python -u train_model1.py --config "configs/$RUN.yaml" --seed 0 --run-name "$RUN" "${resume[@]}" 2>&1 | tee -a "logs/$RUN.log"
status=${PIPESTATUS[0]}
log "$RUN exited with status $status"
[ "$status" -eq 0 ] || exit "$status"

for attempt in 1 2 3 4 5; do
  rsync -a -e "$P1" "checkpoints/$RUN/" "root@$P1HOST:/workspace/pacworld/checkpoints/$RUN/" \
    && rsync -a -e "$P1" "outputs/$RUN/" "root@$P1HOST:/workspace/pacworld/outputs/$RUN/" \
    && rsync -a -e "$P1" "logs/$RUN.log" "logs/sync_$RUN.log" "root@$P1HOST:/workspace/pacworld/logs/" && break
  log "rsync attempt $attempt failed; retrying in 60 s"; sleep 60
done
here=$(sha256sum "checkpoints/$RUN/model1_ema.pt" | cut -d' ' -f1)
there=$($P1 "root@$P1HOST" "sha256sum /workspace/pacworld/checkpoints/$RUN/model1_ema.pt" | cut -d' ' -f1)
if [ "$here" != "$there" ]; then log "checksum mismatch after sync ($here vs $there); pod left running"; exit 2; fi
log "synced checkpoints/$RUN, outputs/$RUN and logs to pod 1 (ema sha256 $here)"
$P1 "root@$P1HOST" "touch /workspace/pacworld/logs/$RUN.done" || true

if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  log "stopping pod $RUNPOD_POD_ID"
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1   # the pod's runpodctl reads the key from its config file
  runpodctl stop pod "$RUNPOD_POD_ID" 2>&1 | tee -a "logs/sync_$RUN.log"
else
  log "RUNPOD_API_KEY / RUNPOD_POD_ID not set: stop this pod by hand (runpodctl stop pod <id> or the RunPod console)"
fi
