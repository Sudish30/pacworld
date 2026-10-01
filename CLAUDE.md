# pacworld

A playable neural world model of Ms. Pac-Man: an action-conditioned diffusion
model in the style of DIAMOND that predicts the next frame given past frames
and the player's action, so the game can be "played" inside the model.

## How to work with me

- I am learning this system. Before implementing anything non-trivial,
  explain the design in a few sentences and wait for my OK.
- After finishing a piece of work, offer a walkthrough of what was built.

## Workflow (2026-10-01; replaces "wait for my OK" above while it stands)

- **Reviewer.** `.claude/agents/research-reviewer.md` defines a skeptical reviewer agent. It is run BEFORE any GPU
  spend, before committing any pre-registration, before any change to README/paper claims, before any push, and
  after any gate result. Its verdict (APPROVE / APPROVE WITH CHANGES) decides; there is no escalation to the owner.
  It never approves retuning a failed pre-registered test.
- **Pre-approved, no need to ask the owner:**
  - pushing to GitHub `main` after the reviewer approves;
  - opening or closing lines of work on the reviewer's verdict;
  - GPU spend up to **$40 more from 2026-10-01 (balance then $21.01)**, keeping the RunPod balance **above $3**;
  - deleting stopped pods that hold no data.
- **Hard limits. Never:**
  - delete or modify the network volume or its data;
  - rewrite git history or force-push;
  - exceed the spend cap or let the balance hit zero;
  - create, change or expose credentials or API keys;
  - submit to arXiv, or anything else under the owner's name except pushing this repo. arXiv files are prepared
    and the owner is told when they are ready.

  If one of these would be needed: write it in `notes/owner_updates.md` and move to other work, or stop if nothing
  else is useful.
- **Owner updates.** At each milestone, a short plain-language update goes in `notes/owner_updates.md` (what
  happened, why, what it means, what's next; jargon explained) and is summarised in chat. When money runs out or the
  work is done, say so clearly.

The full rules and the running log are at the top of `notes/handoff.md`.

## Conventions

- All hyperparameters live in `configs/*.yaml`. No magic numbers in scripts.
- Every script takes `--seed`.
- Every training run logs to wandb.

## Environment

- Python venv at `.venv` (Python 3.12). Activate with `source .venv/bin/activate`.
- Env: `ALE/MsPacman-v5` via gymnasium + ale-py. Smoke test: `python smoke_test.py`.
