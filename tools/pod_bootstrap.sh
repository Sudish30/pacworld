#!/usr/bin/env bash
# Prepare a fresh RunPod pod for one 2M-dataset training run, then launch it.
#
#   bash pod_bootstrap.sh <run-name> <commit> <pod1-host> <pod1-ssh-port>
#   e.g. bash pod_bootstrap.sh m1-2M-ctx8 a625e00 213.173.107.231 11398
#
# Assumes: /workspace exists, python3 (3.12) is installed, this pod can ssh to pod 1 as root
# (its key is in pod 1's authorized_keys) and ~/.netrc holds the wandb login (copied from pod 1).
# Steps: clone the repo at <commit>; rsync the RAW data folders from pod 1 (never the 27 GB cache);
# rebuild the cache with the streaming builder and run the window checks for this run's config;
# verify the frozen val split is the committed one; 150-step smoke run; launch the real run in
# tmux `train` through pod_run_and_sync.sh, which syncs the results back to pod 1 and stops the pod.
set -euo pipefail
RUN=$1; COMMIT=$2; P1HOST=$3; P1PORT=$4
P1="ssh -p $P1PORT -o StrictHostKeyChecking=accept-new"
cd /workspace
if [ ! -d pacworld/.git ]; then git clone https://github.com/Sudish30/pacworld.git pacworld; fi
cd pacworld
git fetch -q origin && git checkout -q "$COMMIT"
echo "[bootstrap] repo at $(git rev-parse --short HEAD) ($(git log -1 --format=%s))"
[ -f "configs/$RUN.yaml" ] || { echo "no configs/$RUN.yaml"; exit 1; }

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt 2>&1 | tail -2
fi
.venv/bin/python -c "import torch; print('[bootstrap] torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /workspace | tail -1

echo "[bootstrap] rsync raw data from pod 1 (data/agent and data/2m/agent only)"
mkdir -p data/agent data/2m/agent logs
rsync -a --info=progress2 -e "$P1" root@$P1HOST:/workspace/pacworld/data/agent/ data/agent/
rsync -a --info=progress2 -e "$P1" root@$P1HOST:/workspace/pacworld/data/2m/agent/ data/2m/agent/
echo "[bootstrap] episodes: $(ls data/agent | wc -l) original + $(ls data/2m/agent | wc -l) new"
[ "$(ls data/agent | wc -l)" -eq 352 ] && [ "$(ls data/2m/agent | wc -l)" -eq 3527 ] || { echo "episode counts differ from pod 1 (352 + 3527)"; exit 1; }

echo "[bootstrap] frozen val split check"
git log -1 --format="  configs/val_episodes_2m.json last changed in %h: %s" -- configs/val_episodes_2m.json
[ "$(git log -1 --format=%h -- configs/val_episodes_2m.json)" = "a7ea0d5" ] || { echo "val split file is not the frozen a7ea0d5 version"; exit 1; }
sha256sum configs/val_episodes_2m.json
$P1 root@$P1HOST "sha256sum /workspace/pacworld/configs/val_episodes_2m.json" | cut -d' ' -f1 > /tmp/p1.sha
[ "$(sha256sum configs/val_episodes_2m.json | cut -d' ' -f1)" = "$(cat /tmp/p1.sha)" ] || { echo "val split differs from pod 1"; exit 1; }
grep -E "val_episodes|folder|cache|context" "configs/$RUN.yaml"

echo "[bootstrap] build cache + window checks"
.venv/bin/python -u dataset.py --config "configs/$RUN.yaml" --seed 0 --rebuild --workers "$(nproc)" --check-windows 12 2>&1 | grep -v "^  cached" | tee logs/cache_2m.log
grep -q "history buffer == dataset window" logs/cache_2m.log

echo "[bootstrap] smoke run (150 steps, wandb disabled)"
.venv/bin/python -u train_model1.py --config "configs/$RUN.yaml" --seed 0 --steps 150 --eval-every 150 --wandb-mode disabled --run-name "smoke-$RUN" 2>&1 | grep -E "step |eval @|done:|params|Traceback|Error"
rm -rf "checkpoints/$RUN" "outputs/$RUN"

echo "[bootstrap] wandb login: $(.venv/bin/wandb login --verify 2>&1 | tail -1)"
tmux new -d -s train "bash tools/pod_run_and_sync.sh $RUN $P1HOST $P1PORT"
sleep 60
tail -n 3 "logs/$RUN.log"
echo "[bootstrap] launched $RUN in tmux 'train'"
