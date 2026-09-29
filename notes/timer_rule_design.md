# The hidden-timer rule: design and pre-registration (2026-09-29)

**Status:** PRE-REGISTRATION. Committing this file fixes the predictions and pass/fail criteria before any run exists.
Nothing below may change after results are seen; a criterion that turns out to be wrong is reported, not edited.

## 1. The claim under test

In pacworld, the pen timer shows the pattern. Models with a 4- or 8-frame consecutive context "park" ghosts in the pen
(16/30 and 23/30 rollouts parked 200+ steps), while a strided context reaching 97 steps releases them on time (0/30).
The real pen stay is at most 91 steps. The frightened timer (124-134 steps) was *not* fixed by reach alone (r148,
reach 145). That shows reach is not sufficient when the event is rare.

**Rule.** An autoregressive frame model that only sees its context can time a hidden-timer event of period N only as
precisely as its context offsets resolve −N.
- If −N is one of its offsets, timing is exact.
- If −N falls between two offsets inside the reach R, the timing is spread over that gap.
- If N > R, the timing is lost. The best possible behaviour is then a constant hazard: a geometric wait with the right
  mean and a spread of about N − R. A sampling model may do worse and park.
- A strided context moves the failure point from N ≈ K (frame count) to N ≈ R (reach), at the price of gaps.

## 2. The synthetic game (`timer_game.py`, no actions, 16x16 RGB, exact palette)

- **Arena.** A 1-px wall border and a pen box at the top centre, whose interior is a 2x2 cell. Below the pen is a
  roaming area.
- **Player.** A 2x2 yellow sprite on a random walk with inertia (keeps its direction with probability 0.8), 1 px per
  step, in the roaming area. It makes frames non-static, and its path carries no information about time.
- **Ghost.**
  - A 2x2 red sprite that roams like the player for D ~ Uniform{10..30} steps.
  - It is then *captured*: it jumps into the pen, and the capture frame c is the first frame showing it there.
  - It stays exactly N frames (c .. c+N-1) and reappears just below the pen at frame c+N (the release), then roams again.
  - The pen shows no countdown, and nothing on screen changes with elapsed time. The only clock is the capture frame
    itself.
- **Data per N:** 60 train episodes and 20 test episodes of 3,000 steps. Seeds: train 1000+e, test 9000+e. Each
  generated set is recorded with a sha256. A new game seed never overlaps a test seed.

## 3. Models and the one variable

- Every model is the pacworld EDM diffusion UNet (`model1.py`, same preconditioning, sampler and context-noise
  augmentation), made small and unconditional (widths [64, 128], attention at 8 px, `n_actions` 1).
- Training: 15,000 steps, batch 128, lr 1e-4 with 100 warm-up steps and cosine decay to 1e-5, seed 0.
- **The only variable between layouts is the context offsets:**
  - **C4**: [-4..-1], reach 4 (Model 1's layout);
  - **C10**: [-10..-1], reach 10;
  - **S10**: [-97,-81,-65,-49,-33,-17,-4,-3,-2,-1], reach 97 (ctx6s16's layout; same frame count as C10).
- **Grid of N:** C4 {3, 8, 17, 33, 65}; C10 {3, 8, 10, 17, 33, 65, 97, 200}; S10 {3, 8, 17, 24, 33, 65, 80, 97, 145, 200}.
  That is 23 models. Every training run logs to wandb project `pacworld` (group `timer-rule`).

## 4. Measurement (`eval/timer_eval.py`)

- **Rollouts.** From every capture in the test episodes that has at least 100 steps of real history and room for
  the horizon, up to 3 per episode, so up to 60 starts. Each starts with the real frames up to and including the
  capture frame. Then 3-step Euler, context sigma 0, one sampler seed per start, horizon 2N + 50 steps.
- **Detector.** Nearest-palette colour per pixel.
  - "In pen": at least 2 of the 4 pen-interior pixels are ghost red.
  - "Released": a ghost-red pixel outside the pen while the pen interior has fewer than 2.
- **Lag.** The first rollout frame, counted from c, at which the ghost is released. A rollout that never releases by
  the horizon is *parked*.
- **Metrics per (layout, N):**
  - on-time fraction P(|lag − N| ≤ tol), with parked counted as not on time;
  - median |lag − N|, with parked counted as the horizon;
  - release fraction;
  - early-release fraction (lag < N − tol).
- **Bootstrap 95% CIs** use 2,000 resamples over test episodes; all starts of an episode stay together.
- **Detector check** before any model is scored: on the real test frames the detector must give lag = N for 100%
  of starts.

## 5. Predictions: the ideal-observer curve (computed before any model is trained)

The rule's exact prediction for every cell comes from an **ideal observer limited to the layout's offsets**
(`timer_game.ideal_observer`).
- It sees only the ghost's state (penned or not) in the frames at the layout's offsets.
- It releases with the probability P(release next | that pattern), estimated by counting in the *training* episodes
  of the cell.
- It is rolled out from the same test starts as the models, with 200 Monte Carlo draws per start.

This is the best any model with those offsets can do, because the ghost state at the offsets is all the game reveals
about elapsed time. Its on-time fractions, median errors and release fractions are computed and written to
`notes/handoff.md` before the first model is trained.

What it looks like, per layout:
- close to 1 where −N is an offset inside the reach, except where a far offset can fall inside the ghost's *previous*
  pen stay and make a short stay look like a long one;
- spread across the gap inside a gap;
- beyond the reach, a geometric wait with hazard h = 1/(N − R). Analytically,
  lag = R + G with G ~ Geometric(h), so p ≈ (1 − h)^(floor(N−R−tol)−1) − (1 − h)^floor(N−R+tol).
  With tol = max(2, 0.1N):

  | layout | N: p |
  |---|---|
  | C4 | 8: 0.572, 17: 0.148, 33: 0.105, 65: 0.086 |
  | C10 | 17: 0.290, 33: 0.134, 65: 0.096, 97: 0.086, 200: 0.080 |
  | S10 | 145: 0.239 (transition zone, reported only), 200: 0.148 |

A model that parks scores below the observer. The simulated observer replaces these analytic values in the criteria.

## 6. Pass/fail criteria (pre-registered)

Cells are classified before training.
- "On-offset in reach": −N is an offset and N ≤ R.
- "Gap": R_recent < N < R and −N is not an offset.
- "Out of reach": N ≥ 1.5R.
- N between R and 1.5R is a transition zone (only S10 at N=145) and is reported, not scored.

`ideal` below means the simulated ideal observer's value for the same cell, starts and tolerance.

| id | cells | pass |
|---|---|---|
| R1 | on-offset in reach: C4 {3}; C10 {3, 8, 10}; S10 {3, 17, 33, 65, 97} | on-time fraction (tol 2) ≥ min(0.80, ideal − 0.10) |
| R2 | gaps: S10 {8, 24, 80} | release fraction ≥ 0.90 and median \|lag − N\| ≤ ideal median + 3 |
| R3 | out of reach: C4 {8, 17, 33, 65}; C10 {17, 33, 65, 97, 200}; S10 {200} | on-time fraction (tol max(2, 0.1N)) ≤ ideal + 0.15 (no better than an observer that cannot see the capture) |
| R4 | strided fixes long periods: N ∈ {33, 65, 97} | S10 passes R1 **and** C10 passes R3 at the same N |
| D0 | detector on real test frames | lag = N for 100% of starts in every cell |
| I0 | the ideal observer on real history | computed and committed to the handoff before any model of that cell trains |

- **The rule HOLDS** if D0 passes and every scored cell passes R1-R4.
- **It holds PARTIALLY** if all out-of-reach cells pass R3 but some in-reach cells fail R1 or R2. Reach is then
  necessary but not sufficient at this budget.
- **It FAILS** if any out-of-reach cell beats the blind observer (fails R3). That would mean a model can time periods
  longer than its reach, or that the game leaks a clock.
- Point estimates decide the verdict; bootstrap CIs over test episodes are reported with every number.

### Amendment 1 (2026-09-29, before any model was trained; the reason is recorded in `notes/handoff.md`)

The ideal observer (I0, table in `notes/handoff.md`) itself releases in only 0.891 of starts at S10 N=24 and 0.594
at S10 N=80. After it overshoots a gap, it meets context patterns that never occur in training, and there it parks.
R2's absolute release bar (≥ 0.90) would therefore fail a perfect model in both gap cells. That makes it a wrong
criterion, found from the prediction and not from any result.

R2 now reads: **release fraction ≥ min(0.90, ideal − 0.10) and median |lag − N| ≤ ideal median + 3.**

Nothing else changes. Also noted, unchanged: at S10 N=97 the observer is on time (tol 2) in only 0.299 of starts,
because far offsets fall inside the previous pen stay. R1's bar there is therefore min(0.80, 0.199) = 0.199, a weak
test of that cell.

## 7. What we conclude if the rule does not hold

- **R3 fails.** Either the game leaks elapsed time (checked first: a pixel-level audit of frames c+1..c+N−1, whose
  only differences must be the player sprite), or the diffusion model carries timing through something we did not
  model. Then the pacworld pen-timer story ("reach fixed it") needs another explanation, and the paper says so.
- **R1 fails in reach** (for example S10 at 65 or 97). A far frame showing the capture is not enough for this model
  and budget to learn the timing. That matches pacworld's frightened-timer negative (event rarity, a weak loss
  signal), and the paper states the rule as "reach is necessary, not sufficient". A follow-up would vary the number
  of events per step, pre-registered separately.
- **R2 fails.** Gaps hurt more than the offset spacing predicts (for example parking inside a gap). That argues for
  denser far offsets, and the paper reports it as a cost of striding.

## 8. Compute and cost

- 23 small models x 15,000 steps, plus about 1,300 short rollouts per model. Data generation and the detector run on
  the CPU.
- On one RTX 4090 the models are tiny and launch-bound; my estimate is **under 3 GPU-hours, about $2-3**.
- A CPU pilot on the Mac validates the pipeline first (one C10 and one S10 cell at reduced steps). Pilot numbers are
  pipeline checks, not results, and are not scored.

## 9. Part b: real ALE games (a separate pre-registration, written before those runs)

Part b takes 1-2 ALE games whose hidden timer is readable from RAM (for evaluation only), preferably with different
periods. It trains the C10 vs S10 layouts with the pacworld recipe at 64px and tests R1/R3 on the real timer. The
games, RAM addresses, periods, detectors and criteria are chosen and committed in `notes/handoff.md` before any data
is recorded.
