# pacworld

A playable neural world model of Ms. Pac-Man: an action-conditioned diffusion model in the style of
[DIAMOND](https://arxiv.org/abs/2405.12399) that predicts the next frame from the last few frames and the
player's action. There is no game engine underneath — you play inside the model, in the browser, at 15 fps.

<!-- DEMO GIF SLOT: drop the recording in docs/demo.gif and uncomment.
     Training also writes a short rollout beside ground truth to outputs/<run>/rollout_<step>.gif.
![playing inside the model](docs/demo.gif)
-->

## What is here

The model is a 19M-parameter EDM diffusion UNet over the noisy next frame plus its context frames, conditioned on
the actions, at 64x64. Everything else in the repo exists to answer one question: *how far can you walk around
inside it before the world stops making sense?*

The interesting finding is that the failures are **hidden timers**. Ms. Pac-Man runs clocks the screen does not
show: how long ghosts stay in the pen after a death (up to 91 steps), how long a power pellet lasts (124-134
steps). A model that sees only the last 4 frames cannot observe them, so it parks the ghosts in the pen forever.
Giving the context a longer *reach* without more frames — 4 recent frames plus 6 strided 16 steps apart, spanning
97 steps — fixes the pen timer outright. The frightened-phase timer is still open; see
[`notes/handoff.md`](notes/handoff.md) for the full log of what was tried, what was pre-registered, and what failed.

## Results

Rollouts of 450 steps (30 s) on 10 held-out episodes x 3 sampler seeds, all produced by
[`eval/eval_rollouts.py`](eval/eval_rollouts.py). Every number below is read from the CSVs in `eval/results/`.

| model | context | wall IoU | pellet IoU | Pac-Man err (px) | responsiveness | ghosts (of 4.0) |
|---|---|---|---|---|---|---|
| Model 0 (MSE baseline) | 4 frames | 0.25 <sub>(900 steps)</sub> | - | - | - | - |
| Model 1 | 4 consecutive | 0.958 | 0.839 | 19.5 | 0.30 | 1.65 |
| + 10x data | 4 consecutive | 0.959 | 0.848 | 19.0 | 0.37 | 2.00 |
| + 10x data | 8 consecutive | 0.960 | 0.829 | 19.1 | 0.38 | 1.47 |
| **+ 10x data, strided** | **4 + 6 @ stride 16** | 0.952 | 0.843 | 13.7 | 0.38 | 2.03 |
| **+ LR anneal** (served) | 4 + 6 @ stride 16 | 0.955 | 0.845 | 14.6 | 0.43 | 2.32 |
| at 128x128 | 4 + 6 @ stride 16 | 0.970 | 0.831 | 21.9† | 0.26 | 2.04 |

The pen timer, measured as how long the ghost pen stays occupied (`eval/pen_timer_analysis.py`):

| | longest pen stay, median | max | rollouts parked 200+ steps |
|---|---|---|---|
| real game | 78 | 91 | 0 / 30 |
| Model 1 (4 frames) | 222 | 373 | 16 / 30 |
| 8 consecutive frames | 324 | 381 | 23 / 30 |
| **4 + 6 strided (reach 97)** | **82** | 148 | **0 / 30** |

† at 64x64-equivalent (43.8 px on its own 128px frames). The 128px model is sharper per step — Pac-Man error at
15 steps is 0.12 px vs 1.90 at 64px — but drifts faster over a long rollout, so the 64px model is the one served.
Its pen numbers are not comparable across resolutions: the occupancy measure is not resolution-invariant
(ground truth is 78 steps at 64px but 38 at 128px). Details in `notes/handoff.md`.

## Repository layout

```
record.py            record episodes with a PPO agent (+ eps/sticky exploration, or a power-pellet-seeking
                     policy) into data/<mode>/ep_<seed>.npz: frames, actions, rewards, RAM, seed
playback.py          replay a recording as a GIF; --check-alignment re-simulates it to prove frames and
                     actions line up with the emulator
dataset.py           the frame cache and the windowed dataset. gather_context() is the single definition of
                     "what the model sees": context offsets, clamping at an episode start, paired actions.
                     History replays the same rule during rollouts, so play matches training
model1.py            the EDM diffusion UNet (Karras et al. 2022 preconditioning) + Euler sampler
model0.py            a plain MSE next-frame UNet, as a baseline
train_model1.py      training loop: EMA, context-noise augmentation, periodic val loss / sample grid /
                     rollout GIF, checkpoints, wandb
serve/               the playable demo: FastAPI + WebSocket server, browser canvas, a watchdog that starts a
                     fresh board when the model loses Pac-Man
eval/                the measurement stack. detectors.py finds sprites, pellets and walls in a generated
                     frame and is validated against native-resolution truth; eval_rollouts.py scores
                     rollouts; the rest diagnose specific failures (pen timer, ghost loss, drift, freezes)
tools/               data and infrastructure helpers: event indexing, val-split freezing, maze graph,
                     cache verification, benchmarks, pod orchestration scripts
configs/             every hyperparameter, one YAML per run. No magic numbers in scripts
notes/handoff.md     the running lab notebook: findings, pre-registered predictions and their verdicts
```

`data/`, `checkpoints/`, `outputs/`, `logs/` and `eval/results/` are generated and not tracked.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate    # Python 3.12
pip install -r requirements.txt
python smoke_test.py                                 # checks ALE/MsPacman-v5 runs
```

Every script takes `--seed`, reads its hyperparameters from a config in `configs/`, and every training run logs
to wandb (`--wandb-mode disabled` to turn that off).

**1. Record.** Writes one `.npz` per episode; 200k steps takes a few minutes on CPU.

```bash
python record.py --seed 42 --mode agent --steps 200000          # -> data/agent/
python playback.py --seed 7 --mode agent --check-alignment      # verify the recording
```

**2. Build the frame cache** (resize, split off the frozen validation episodes, check every window):

```bash
python dataset.py --config configs/model1.yaml --seed 0 --rebuild --check-windows 12
```

**3. Train.** `configs/model1.yaml` is the 4-frame baseline; `configs/m1-2M-ctx6s16.yaml` is the strided-context
recipe, and `configs/m1-2M-ctx6s16-ft-uniform.yaml` the short low-LR anneal that follows it.

```bash
python train_model1.py --config configs/m1-2M-ctx6s16.yaml --seed 0 --run-name ctx6s16
python train_model1.py --config configs/m1-2M-ctx6s16-ft-uniform.yaml --seed 0 --run-name ctx6s16-ft
```

**4. Evaluate.** Validate the detectors on real frames first, then score rollouts and diagnose:

```bash
python eval/detectors.py   --config configs/eval_m1-2M-ctx6s16-ft-uniform.yaml --seed 0 --episodes 8
python eval/eval_rollouts.py --config configs/eval_m1-2M-ctx6s16-ft-uniform.yaml --seed 0 --model model1 --save-all-preds
python eval/pen_timer_analysis.py --config configs/eval_m1-2M-ctx6s16-ft-uniform.yaml --seed 0
python eval/fair_ghosts.py        --config configs/eval_m1-2M-ctx6s16-ft-uniform.yaml --seed 0
python eval/compare_runs.py --seed 0 --runs model1=eval/results m1-2M-ctx6s16-ft-uniform=eval/results/m1-2M-ctx6s16-ft-uniform
```

**5. Play it.**

```bash
python serve/server.py --seed 0        # configs/serve.yaml picks the checkpoint
```

Then open <http://localhost:8000> and use the arrow keys (or WASD; hold two for diagonals). The server binds to
localhost, so on a remote machine reach it through an SSH tunnel:
`ssh -N -L 8000:localhost:8000 <user>@<host>`. One frame is three Euler sampler steps in bf16, about 28 ms on an
RTX 4090, which holds the 15 fps the game was recorded at.

## Requirements

A CUDA GPU for training (the runs here are on a single RTX 4090; 100k steps at 64x64 takes about 3.3 h, and the
128x128 variant about 8 h). Recording, playback and the detectors are CPU-only. `requirements.txt` pins nothing;
it was developed against PyTorch 2.14 + CUDA 13, gymnasium with `ale-py`, and Python 3.12.

## License

MIT — see [LICENSE](LICENSE).
