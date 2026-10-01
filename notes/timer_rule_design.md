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

## 9. Part b: a real Atari game. PRE-REGISTRATION (2026-09-30, before any data is recorded or any model trained)

**Game search** (exploratory, `notes/handoff.md`). 45+ ALE games were scanned for regular RAM countdowns, and the
frames around each countdown's end were inspected. One clean hidden timer passed every check:
- **ALE/Pacman-v5** (the 2600 Pac-Man, a different game and engine from Ms. Pac-Man).
- After every life loss, one ghost sits in the central house for **exactly 35 or 36 steps** (at our 4-frame skip) and
  then leaves: 196 of 196 stays, 129 x 35 and 68 x 36.
- **RAM byte 100** counts down monotonically through every stay and reads 0 at every release. It is used for
  evaluation only.
- Nothing on screen counts the stay: outside the house the screen repeats a fixed 2-step flicker pattern, and the
  house shows the same motionless ghost throughout. The stay starts at a visible event (the respawn).
- Rejected candidates and why are listed in the handoff.
- No second clean timer with a different period was found. The real-game out-of-reach data point stays Ms.
  Pac-Man's own frightened timer (124-134 steps, beyond S10's 97, never timed by any layout; handoff verdicts).
- The period N ≈ 35 lies beyond C10's reach (10) and inside S10's (97), in S10's gap between offsets −33 and −49.
  The rule predicts: **S10 releases the ghost with timing about as good as an observer limited to its offsets; C10
  does no better than an observer blind to the respawn, and may park.**

**Data** (`configs/record_pacman.yaml`, `record.py --mode random`):
- The recorder's settings otherwise: frameskip 1 with a manual 4-frame skip, max-pool over all 4 frames, maze crop
  rows 0-172, uniform random actions over the 5-action set, repeat_action_probability 0.
- 1,000,000 steps, seed 45.
- Frozen val split: 10% of episodes, chosen once (seed 0), saved in `configs/val_episodes_pacman.json` with its sha256.
- 64px cache built as for Ms. Pac-Man (box resize).

**Models** (the one variable is the context offsets):
- The pacworld recipe: `model1.py`, 18.8M params, 64px, action-conditioned (5 actions), 100k steps at 1e-4 then a
  15k-step anneal at 1e-5, batch 64, seed 0, the same EDM and context-noise settings.
- `pac-C10`: offsets −10..−1. `pac-S10`: ctx6s16's offsets. Own checkpoint folders; every run logged to wandb.

**Measurement** (`eval/pacman_house.py`, evaluation reads RAM):
- Starts: every stay in the val episodes with ≥ 100 steps of real history. Context = real frames up to and including
  the first frame with the house occupied (c).
- Rollouts: recorded actions, 3-step Euler, context sigma 0, horizon 2 x 36 + 50 = 122 steps, one sampler seed per
  start.
- Lag: the first step at which the house reads empty in two consecutive frames. Parked = never within the horizon.
- The truth per start is the RAM release (35 or 36).
- Occupancy is a pixel test at 64px (the house box against the empty-house reference), calibrated only against RAM on
  training episodes.
- Metrics: on-time fraction (|lag − truth| ≤ tol, tol = max(2, 0.1 x 36) = 3.6), median |lag − truth|, release
  fraction. Bootstrap 95% CIs over val episodes.
- The **ideal observer** (house-occupied bits at each layout's offsets, hazard from the training episodes) is computed
  and committed before any model trains, as in part a.

**Pass/fail** (the same logic as part a):

| id | check | pass |
|---|---|---|
| P-D0 | detector on real val frames: the pixel lag equals the RAM release for every start | 100% of starts |
| P-I0 | the ideal observer for both layouts | computed and committed before training |
| P-S | pac-S10 (N inside reach; a gap cell) | release fraction ≥ min(0.90, ideal − 0.10) and median \|lag − truth\| ≤ ideal median + 3 |
| P-C | pac-C10 (N ≥ 1.5 x reach) | on-time fraction ≤ ideal + 0.15 |
| P-R | the rule holds on this game | P-D0, P-S and P-C all pass |

- Reported, not gating: C10's parking rate, early releases, standard rollout metrics.
- **Pre-spend check.** (a) Most likely failures:
  - the house state is not readable at 64px (box blending), which invalidates P-D0;
  - random play gives short episodes, so S10's far offsets often clamp to the episode start;
  - both models lose ghosts altogether (as DIAMOND did), making release timing unmeasurable for reasons unrelated to
    reach.
- **(b) Cheapest tests, run first:**
  - $0: a 50k-step sample, then stay statistics, episode lengths and P-D0 on its 64px frames;
  - GPU pilot: pac-S10 for 20k steps, scored on the val starts. It must render the occupied house and release it
    within the horizon in ≥ 50% of starts, or the project stops before the full runs.
- **(c) Differences from the base recipe** (pacworld on Ms. Pac-Man) and their risks:
  - another game with a 5-action set;
  - random play instead of PPO + ε, which gives less purposeful movement and more deaths. Neutral for a timer that
    runs after deaths, but it changes the data distribution;
  - 1M instead of 2.2M frames: less data, and the pilot checks learnability;
  - a different GPU: speed only.
- **Cost.** $0 recording on the pod CPU. On the RTX 2000 Ada ($0.24/h) the 64px UNet's speed is not measured yet; at
  about 3 it/s the two 115k-step runs are about 21 GPU-hours, **about $5**.

### Part b, Amendment 1 (2026-09-30; from a $0 50k-step sample, before any real data, model or GPU spend)
The pre-spend cheap test (50k steps, seed 45, the registered recording policy) showed that **RAM byte 100 is not the
release timer**.
- It is a countdown (2 per step) that restarts when the house fills and reaches 0 about 30 steps in. The ghost leaves
  5 or 6 steps later.
- The scan's "0 at every release" held only because the byte then sits at 0.
- Measured against it, P-D0 would read 0/312. The criterion was wrong, not the detector.

**Corrections** (nothing else changes; the criteria P-S, P-C and P-R and their bars stay as registered):
- **Truth per start** = the release in the real frames at native resolution: the first of two consecutive frames
  without ghost pixels in the house box. In the sample every genuine stay lasts 35 or 36 steps (191 x 35, 119 x 36).
- RAM byte 100 is reported only as evidence that the hidden timer lives in game memory.
- **Starts** = house-occupancy onsets whose real occupancy lasts at least 10 steps, with ≥ 100 steps of real history.
  Ghosts passing through the house box last at most 5 steps: 2 in the sample.
- **P-D0** = the 64px detector's release equals that truth for 100% of starts. Sample: 310/310.
- Also from the sample: episodes last a median 420 steps (p10 348); 56% of stays begin ≥ 100 steps into their
  episode.

## 10. Part a follow-up: does a frame exactly at −N fix the N=80 failure? PRE-REGISTRATION (2026-10-01, before any follow-up model trains)

**Question.** In part a, S10 at N=80 parked in 37 of 60 starts. S10 has frames at −81 and −65, so −80 falls between
two frames. The owner's explanation: periods on a frame are timed, periods between frames park.

**One variable.** A new layout **S10b** = S10 with the frame at −81 moved to −80:
`[-97, -80, -65, -49, -33, -17, -4, -3, -2, -1]`. Same reach (97), same number of frames, same game, model, 15k
steps, seed 0, test starts and scoring as part a. ("More steps" is a different variable and is not tested.)

**Cells** (`followup_grid` in `configs/timer.yaml`):
- **N=80**: on a frame in S10b.
- **N=72, 79, 88**: fresh periods between frames of S10b. 72 is mid-gap between −80 and −65; 79 is at the far end of
  that gap, one step short of the −80 frame (the analogue of N=80 in S10); 88 is mid-gap between −97 and −80.

**D0 and I0, computed before any model trains** (`eval/timer_eval.py --ideal-only`; D0 passes in all four):

| cell | ideal on-time (tol 2) | ideal release | ideal median \|lag − N\| | bar |
|---|---|---|---|---|
| S10b N=80 (on-frame) | 0.716 | 1.000 | 0 | R1: on-time ≥ min(0.80, 0.716 − 0.10) = **0.616** |
| S10b N=72 (off-frame) | 0.317 | 0.898 | 4 | R2: release ≥ **0.798** and median ≤ 7 |
| S10b N=79 (off-frame) | 0.120 | 0.594 | 12 | R2: release ≥ **0.494** and median ≤ 15 |
| S10b N=88 (off-frame) | 0.254 | 0.874 | 5 | R2: release ≥ **0.774** and median ≤ 8 |

**Scoring: as part a** (`tools/timer_followup_verdict.py`, written before any result). R1 for the on-frame cell, R2
for the off-frame cells. A cell **parks** if its release fraction is below its R2 release bar. Point estimates decide;
bootstrap CIs over test episodes are reported.

**The owner's prediction, committed before running: the on-frame period passes, the off-frame periods park.**
- **SUPPORTED**: N=80 passes R1 and all three off-frame cells park.
- **PARTLY SUPPORTED**: N=80 passes R1 and one or two off-frame cells park.
- **ON-FRAME ONLY**: N=80 passes R1 and no off-frame cell parks.
- **NOT SUPPORTED**: N=80 fails R1. Then the missing frame was not what broke N=80 in S10.

**Where the ideal observer disagrees with that prediction (noted before running).**
- The observer does not park in every gap. It releases in about 0.9 of starts mid-gap (N=72, 88) and in only 0.59 at
  the far end of a gap (N=79), because there it has no room to overshoot.
- Part a already showed two gap cells that did not park (S10 at N=8 and N=24, both passing R2).
- So the observer predicts "N=79 at risk, N=72 and 88 release", and the owner's prediction is "all three park". The
  mid-gap cells separate the two.

**Pre-spend check.**
- (a) Most likely failures: N=80 still parks with a frame at −80 (then far frames are simply hard at this budget, as
  at N=97); or the mid-gap cells release, which contradicts the prediction but agrees with the observer.
- (b) Cheapest test: the pipeline and both layout families already ran in part a; D0 and I0 above cost $0. No smaller
  GPU test exists than the four cells themselves.
- (c) Differences from part a: the layout only.
- **Cost:** 4 cells x 15k steps on the RTX 2000 Ada ($0.24/h), about 1.5 GPU-hours, **under $0.50**.

## 11. Part b, attempt 2. PRE-REGISTRATION (2026-10-01, before any attempt-2 training)

**History.** Attempt 1 stopped at its registered pilot: pac-S10 at 20k steps released in 0.486 of 622 val starts,
below the 0.50 bar. That is reported as a stopped pilot and is not re-scored. No part-b criterion (P-S, P-C, P-R) has
been scored.

**What attempt 2 is** (the owner's decision): the full runs exactly as designed in section 9 + Amendment 1, with the
same data, split, models, seeds, measurement and bars. The only difference from attempt 1 is that the 20k-step pilot
rule is not applied again.
- **pac-S10** continues from the pilot's 20k-step checkpoint (`--resume`) to 100k steps, then the 15k-step anneal.
  This is the continuation section 9 planned.
- **pac-C10**: 100k steps, then the 15k-step anneal.

**Bars, unchanged** (numbers from P-I0, committed in d88bff5 / 894acd9):
- **P-S** (pac-S10-ft-uniform): release fraction ≥ **0.898** and median |lag − truth| ≤ **5**.
- **P-C** (pac-C10-ft-uniform): on-time fraction ≤ **0.251**.
- **P-R**: P-D0 (passed, 622/622), P-S and P-C all pass.

**Order and gate.**
1. Train and score pac-S10 first. **P-S is the gate.**
2. If P-S fails: **part b stops for good.** pac-C10 is not trained, and P-R is reported as failed.
3. If P-S passes: train and score pac-C10, then P-C and P-R.

Reported, not gating: pac-S10 scored at 100k steps (before the anneal); C10's parking rate; early releases.

**Pre-spend check.**
- (a) Most likely failure: pac-S10 still parks after 115k steps. At 20k it parked in 51% of starts and was late in
  the rest. Random play gives short episodes (median 420 steps), so 20.8% of training windows have a clamped far
  frame, and the stay sits between frames (−33 and −49) of this layout.
- (b) Cheapest test: the 20k pilot was that test, and it failed. No cheaper test of "does it learn by 115k" exists
  than training S10 first, which is why C10 waits for P-S. That saves about half the cost on a failure.
- (c) Differences from section 9: none in data, model or scoring. The S10 run is resumed once (new random stream for
  batch sampling after the resume), and the pilot gate is dropped.
- **Cost** on an RTX 4090 ($0.74/h, 9.0 it/s measured): S10 needs 95k more steps, about 2.9 GPU-hours, **about $2.2**.
  C10 needs 115k steps, about 3.6 GPU-hours, **about $2.6**, spent only if P-S passes. Total at most **about $5**
  with scoring.
