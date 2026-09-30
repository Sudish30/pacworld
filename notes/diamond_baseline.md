# External baseline: DIAMOND's released Ms. Pac-Man world model (STEP 5), full findings

Pre-registration commit 7293ef0; harness commit 345c890; result recorded in `notes/handoff.md` (commit f5bd555). All
numbers below are read from `eval/results/diamond/{gates,pen}.json` and `eval/results/stats/bootstrap_ci.json`.

## 1. What exists

- **DIAMOND** (Alonso et al., NeurIPS 2024; MIT license) released trained world models for the 26 Atari-100k games:
  Hugging Face `eloialonso/diamond`, `atari_100k/models/MsPacman.pt`, 54,323,485 bytes, sha256 `91b213ef3f9e9091...`.
  We use only its denoiser; the checkpoint also holds a reward/termination model and an actor-critic.
- **Denoiser:** a 4.41M-param UNet (channels [64, 64, 64, 64], no attention), conditioned on **4 previous frames and
  4 actions** (9 actions, the same minimal set as ours).
  - EDM preconditioning with sigma_offset_noise 0.3.
  - Sampled with DIAMOND's own world-model sampler: 3 Euler steps, sigma 2e-3 to 5, rho 7.
- **Preprocessing:** the full 210x160 screen (score area included), max over the last **2** of the 4 skipped frames,
  cv2 area resize to 64x64 RGB.
- **Training:** 100k agent steps (Atari-100k), with episodes ending at every life loss. The world model was built for
  **15-step** imagination rollouts during RL.
- I found no other released, trained Ms. Pac-Man pixel world model. DIAMOND is the directly comparable one; this
  search did not cover everything.

## 2. What I ran (no retraining; Mac CPU, $0)

- **Episodes:** the same 10 held-out episodes (the 10 longest of `configs/val_episodes.json`) x 3 sampler seeds,
  start step 100, 900 steps, the recorded actions.
- **DIAMOND-format frames of the same game states:** each episode re-simulated from its seed and actions in ALE,
  capturing DIAMOND's preprocessing alongside ours.
- **Validity gates on ground truth, all passed:**
  - V-D1: the re-simulation reproduces our recorded frames bit for bit in all 10 episodes.
  - V-D2: Pac-Man detected in 99.9% of gated ground-truth frames.
  - V-D3: the pen measure on ground truth gives a longest-stay median of 94 (band 70-95) and 0/30 parked.
- **Detectors rebuilt for DIAMOND's geometry:** the same maze reference passed through DIAMOND's resize, with
  detection limited to the maze rows. The reference finds the same 152 pellets as ours.
  - Pac-Man positions are converted to our 64px-equivalent coordinates, so Pac-Man error and responsiveness are
    comparable.
  - Pen occupancy uses the pen box mapped to DIAMOND's rows, "occupied in any of the last 4 frames", because 2-frame
    pooling makes penned ghosts flicker.
- **Code:** `eval/diamond_baseline.py`, `configs/eval_diamond.yaml`; DIAMOND's source and weights in `external/`
  (git-ignored).

## 3. The pre-registered prediction and its result

**P-D1 (the timer rule):** DIAMOND's 4-consecutive-frame context parks ghosts: the pen occupied 200+ consecutive
steps in **≥ 10/30** rollouts. For comparison, Model 1 had 16/30, ctx4 17/30, the real game 0/30.
- Pre-stated escape: if DIAMOND rendered fewer than 5 own respawns in total, P-D1 would be "untestable".

**Result: FAIL as registered.** 0/30 rollouts parked. Longest pen stay median 16 steps (ground truth 94). 6 own
respawns, above the minimum of 5, so the test counts.

**Why:** DIAMOND loses the ghosts entirely, so the pen never refills.

| ghost count / its own ground truth | @15 | @150 | @450 |
|---|---|---|---|
| DIAMOND | 0.976 [0.954, 0.995] | 0.301 [0.257, 0.365] | **0.077 [0.054, 0.097]** |
| our Model 1 (4 frames) | 1.000 | 0.240 [0.125, 0.344] | 0.411 [0.250, 0.609] |
| our ft-uniform (strided, served) | 1.000 | 0.473 [0.250, 0.670] | 0.581 [0.297, 0.865] |

The frames at steps 150-300 show DIAMOND's maze, pellets and Pac-Man intact, with no ghosts.

**My reading:** P-D1 implicitly assumed ghosts survive long enough to be penned, and DIAMOND fails earlier and harder.
The result is **neither support for the timer rule nor evidence against it**: DIAMOND does not time the pen
correctly, it simply has nothing left to pen. The criterion is left as registered.

## 4. Everything else (episode-bootstrap 95% CIs, 10 episodes x 3 seeds, each model against its own ground truth)

| metric | DIAMOND | Model 1 | ft-uniform (served) |
|---|---|---|---|
| Pac-Man error @15 (px, 64px-equiv.) | 2.87 [1.04, 5.36] | 1.79 [0.75, 3.04] | 1.90 [0.74, 3.25] |
| Pac-Man error @150 | 20.58 [16.17, 25.44] | 13.08 [10.23, 16.16] | 10.23 [6.25, 14.50] |
| Pac-Man error @450 | 25.31 [19.07, 31.23]* | 19.48 [13.27, 25.42] | 14.57 [10.36, 19.29] |
| responsiveness @15 | 0.619 [0.541, 0.694] | 0.788 [0.697, 0.887] | 0.799 [0.703, 0.893] |
| responsiveness @450 | 0.183 [0.133, 0.238] | 0.303 [0.228, 0.384] | 0.429 [0.395, 0.464] |
| pellet IoU @450 | 0.781 [0.755, 0.814] | 0.839 [0.821, 0.853] | 0.845 [0.825, 0.864] |
| wall IoU / own ceiling @450 | **1.000** [0.995, 1.004] | 0.976 [0.971, 0.981] | 0.973 [0.968, 0.978] |

\* DIAMOND's mean over rollouts is 28.18. The per-episode mean differs because some rollouts lose Pac-Man, which
gives NaN errors.

Paired differences on the same episodes, DIAMOND − ft-uniform @450:
- Pac-Man error +10.7 [+4.5, +17.4];
- responsiveness −0.246 [−0.305, −0.185];
- pellet IoU −0.064 [−0.094, −0.031];
- wall IoU / ceiling +0.027 [+0.023, +0.030];
- ghost ratio −0.50 [−0.81, −0.20].

DIAMOND keeps the maze walls better than any of our models.

## 5. Caveats that go with any use of these numbers

- 900 steps is 60x DIAMOND's 15-step design horizon. It trained on about 1/20 of our data (100k agent steps vs 2.2M
  frames) and never trained across a life loss.
- The comparison therefore measures long-horizon coherence, which DIAMOND was not built for. It does not rank the two
  designs.
- DIAMOND sees the score area and uses 2-frame pooling (ghosts flicker in its own ground truth). Every metric is scored
  against DIAMOND's own ground truth in its own format, and ratios are used where the formats differ.

## 6. Options for the owner

1. Report P-D1 as a plain **FAIL** in the paper, with the ghost-collapse explanation and the table above.
2. Additionally register a **new, clearly post-hoc** test that conditions on ghost survival, for example parking
   measured only in rollouts that keep ≥ 2 ghosts. It could not undo the registered FAIL.
3. Present DIAMOND only as a coherence baseline (sections 4-5) and discuss P-D1 in the limitations.
