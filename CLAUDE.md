# pacworld

A playable neural world model of Ms. Pac-Man: an action-conditioned diffusion
model in the style of DIAMOND that predicts the next frame given past frames
and the player's action, so the game can be "played" inside the model.

## How to work with me

- I am learning this system. Before implementing anything non-trivial,
  explain the design in a few sentences and wait for my OK.
- After finishing a piece of work, offer a walkthrough of what was built.

## Conventions

- All hyperparameters live in `configs/*.yaml`. No magic numbers in scripts.
- Every script takes `--seed`.
- Every training run logs to wandb.

## Environment

- Python venv at `.venv` (Python 3.12). Activate with `source .venv/bin/activate`.
- Env: `ALE/MsPacman-v5` via gymnasium + ale-py. Smoke test: `python smoke_test.py`.
