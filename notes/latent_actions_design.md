# Learned controls: latent actions for pacworld (design, rev 2, 2026-09-29)

**Status: approved (rev 2, 2026-09-29).** §9 is the pre-registration. It is copied verbatim into `notes/handoff.md`
and committed together with `eval/visible_ceiling.py` and `configs/lam_agreement.yaml` before any Stage 0 code
touches the data. G1 was settled after the ceiling measurement in §3a: lag 1, 0.70.

## 0. Summary

A **latent action model (LAM)** looks at one transition (frames t-2..t and t+1) and has to squeeze it into one of
**8 codes**. A decoder then has to predict frame t+1 from the past plus that code. 3 bits cannot hold a frame, so the
code ends up carrying whatever part of the next frame the past could not predict. The design is set up so that this
part is Pac-Man's move. The LAM is then frozen and labels all 2.2M transitions with codes. The **ctx6s16 diffusion
recipe (100k steps + 15k-step LR anneal) is trained from scratch on those codes instead of actions**. At play time,
arrow keys map to codes through a mapping found from unlabeled frames: a pixel detector reads which way Pac-Man moves
under each code. A small prior network decides when a key's code is possible here, and that same rule defines the
no-op. True actions and RAM are read **only** by evaluation scripts.

**The stop point:** after Stage 1 (LAM only; Stage 0 + 1 ≈ 4.3 pod-hours, ~$3.25), the project stops before any
diffusion training if either of these holds:

- at Pac-Man's turns, the codes (lag 1, §3a) agree with the true action's direction **less than 70%** of the time
  after the best code-to-direction mapping, or less than majority + 0.30 (G1);
- keys sent through the label-free mapping reach a code of the right direction **less than 65%** of the time at turns,
  or less than 85% over all steps (G2).

On a stop: report, no retuning to pass. Only arms A and B exist; any other LAM variant needs its own pre-registration.

The whole plan is about 15 pod-hours (~$12; budget $20 for slow hosts and reruns) and ~5 GB of new disk after
deleting the 128px cache (33.7 GiB).

```
                         ── training uses frames only ──
 frames ─► LAM encoder (t-2..t, t+1) ─► VQ, 8 codes ─► LAM decoder (ctx6s16 past + code) ─► frame t+1    Stage 1
                              │
                 codes for all 2.2M transitions (int8, same shape as the actions array)
                ┌─────────────┴──────────────────────────┐
     prior p(code | past)                  diffusion WM, ctx6s16 recipe, codes in the action slots      Stage 2
                │                                         │
   keys ─► resolver ─► code ─────────────────────────────►  next frame ─► LAM re-labels it ─► History     play

                   ── evaluation only: true actions + RAM ──
   G1/G2 (code-action agreement), junction controllability test, responsiveness, standard rollout metrics  Stages 1, 3
```

## 1. What exists and gets reused

| piece | file | reused as |
|---|---|---|
| context rule, clamping at episode start | `dataset.gather_context` | unchanged. The codes array replaces the actions array, so every window and every rollout step sees codes through the same function. The LAM encoder also gathers its 4 frames with it (offsets `[-3,-2,-1,0]`). |
| training windows | `dataset.WindowDataset` | unchanged; `get_datasets` swaps `cache["actions"]` for codes (or zeros) when `data.action_source` says so |
| rollout buffer | `dataset.History` | unchanged except one small method, `relabel_last(code)`, for re-inference (§7.5). It stores integers and does not care whether they are actions or codes. |
| window checks | `dataset.check_windows` | run again on the codes array (History == dataset window, nothing leaves its episode) |
| diffusion UNet, EDM denoiser, Euler sampler | `model1.py` | WM unchanged (`n_actions: 8`, or `1` for no-action). `model1.UNet` is also the LAM decoder. |
| training loop, EMA, prefetch, GPU noise, fine-tune from `init_from`, wandb | `train_model1.py` | unchanged; the in-training rollout GIF replays whatever `cache["actions"]` holds (codes) |
| frozen split | `configs/val_episodes_2m.json` (388 val episodes, sha256 `7ba75f89...`) | same split for the LAM, the prior, both new WMs and every evaluation |
| sprite / wall / pellet detectors | `eval/detectors.py` | Pac-Man positions for the label-free key mapping and arm B's loss weight. **Pixels only:** the reference maze comes from a native frame, with no RAM or actions. |
| rollout eval, metrics, pen/ghost analyses, comparison | `eval/eval_rollouts.py`, `metrics.py`, `pen_timer_analysis.py`, `fair_ghosts.py`, `compare_runs.py` | same 10 episodes x 3 seeds; one new config key `rollout.action_source` (default `recorded`, so existing configs behave exactly as today) |
| maze graph | `tools/build_maze_graph.py` (mined from RAM) | **eval only**: which directions are legal at a junction, for the junction test |
| demo server | `serve/server.py` | **not edited.** A new `serve/server_lam.py` subclasses `World`, on its own port. |

## 2. The label firewall (making the hard rule mechanical)

- **Training and mapping code never loads labels.** With `data.action_source: lam` (or `none`), `get_datasets`
  replaces `cache["actions"]` with the codes array (or zeros) right after loading the meta file, before any dataset
  object exists. It asserts that the replacement has the same shape and the same `-1` end-of-episode markers. RAM is
  not in the cache at all. It lives only in the raw `.npz` files, which only `eval/` and the recording tools read.
- **Only these scripts read true actions or RAM:** `eval/lam_agreement.py` (G1, G2 and the reported agreement numbers),
  `eval/junction_test.py` (legality from the RAM maze graph, true actions for the labeled baseline),
  `eval/eval_rollouts.py` in `recorded` / `lam_mapped` modes, and the existing pen/ghost analyses (RAM gating).
- **No model selection by labels:**
  - The LAM, prior and WM step counts are fixed in advance and the final weights are used.
  - The LAM arm is chosen by a label-free criterion (§4d), and that choice is committed to the handoff **before**
    `lam_agreement.py` runs.
  - The resolver thresholds are calibrated on unlabeled train transitions (§7.2).
  - The key mapping comes from pixels (§7.1).
  - Labels decide only the go/no-go gates and the final verdict.
- **Nothing is fine-tuned from a labeled checkpoint.** The latent and no-action WMs start from scratch, because the
  ctx6s16 weights were trained on true actions.
- **Tested, not just asserted (Stage 0):**
  - The LAM smoke run is done twice: once as is, once with the cache's true actions array replaced by random integers
    before `get_datasets`. The codes and every logged loss must come out bit-identical. If anything downstream still
    read the labels, the two runs would differ.
  - Window sampling and loss weights in every from-scratch run are built from nothing derived from actions or RAM.
    The WM configs have no `events:` section, so sampling is uniform (`WindowDataset.sample`), and
    `train_model1.py` refuses an `events:` section when `data.action_source` is `lam` or `none`. The context-noise
    draws use only random generators. Arm B's weight map uses only the pixel detector's Pac-Man positions.
- **Honest caveat:** arm B's loss weight and the key mapping use the Pac-Man *detector*. It uses no labels, but it does
  tell the system which sprite is the player. That is domain knowledge Genie does not use. Arm A has none of it except
  in the mapping (§12, decision 1).

## 3. LAM architecture (item 1)

**Transition convention (same as the dataset).** `actions[j]` is taken after `frames[j]` and produces `frames[j+1]`.
The LAM gives `codes[j]` to the transition `frames[j] -> frames[j+1]`. So `codes` has exactly the shape of the cache's
actions array, with `-1` on each episode's last frame, and can be dropped in wherever actions are used today.

**Encoder** (~3M params):
- Input: the frames at offsets `[-3,-2,-1,0]` around target i (that is j-2, j-1, j and j+1, with j = i-1), gathered by
  `gather_context` with the usual clamping. That is 12 channels at 64x64.
- Four stride-2 conv stages (64-128-256-256 channels, GroupNorm + SiLU) down to 4x4x256, then a linear layer to a
  32-d vector.
- It sees only 4 frames. It needs recent motion plus the next frame, and nothing long-range.

**Bottleneck:**
- Vector quantisation with a **codebook of 8 x 32** (Genie also uses 8).
- EMA codebook updates (decay 0.99), commitment loss β = 0.25, straight-through gradient.
- A code unused for 2,000 steps is re-seeded from a random encoder output, which prevents codebook collapse.
- One code per transition, so at most 3 bits.
- Why 8: Pac-Man needs 4 directions, plus room for "stationary", death/reset and spare codes. 16 would invite
  near-duplicate codes (for example "up while turning" vs "up while continuing"). That is harmless for agreement but
  makes the key mapping messier.

**Decoder** (~18.8M params): `model1.UNet` with the model1 widths `[64,96,192,192]` and attention at 16 and 8 px.
- Input: the **ctx6s16 context** (10 frames, offsets `[-97,...,-1]` relative to i; 30 channels).
- One action slot, which receives the code through the existing AdaGN conditioning.
- The noise-level inputs are held constant, so it is a deterministic regression.
- It predicts frame i as a residual on frame i-1, with MSE loss (arm A) or player-weighted MSE (arm B, §4c).
- A deterministic decoder blurs whatever it cannot predict. So the code gets gradient exactly where settling an
  ambiguity removes the most blur.

**Training** (config `configs/lam-{A,B}.yaml`):
- 30k steps, batch 64, AdamW lr 1e-4 with 500 warm-up steps, then cosine down to 1e-5 (the handoff lesson: end every
  run with an LR decay).
- bf16, `--seed 0`, train episodes only.
- wandb runs `lam-A` / `lam-B`, logging: loss, commitment loss, codebook perplexity, code-usage histogram, code gain,
  and a **code-swap grid** (one past decoded with each of the 8 codes, side by side). That grid shows directly what each
  code does.

**How it is kept from encoding the next frame instead of the action:**
1. **Capacity.** 3 bits per transition, against 98,304 bits in a 64x64 RGB frame. It can carry one small decision, not
   a picture.
2. **Only one path.** The future reaches the decoder only through the quantised code. There are no encoder-to-decoder
   skip connections, and the past reaches the decoder straight from the dataset.
3. **Explain away everything predictable.** The decoder sees the same 10-frame, 97-step context that made pen releases
   predictable for the WM. Ghost paths, the pen timer, pellet eating and Pac-Man's chomp cycle then need no code, and
   the 3 bits go to the largest remaining surprise.
4. **Short encoder window.** 4 frames, so the encoder cannot smuggle in long-horizon state the decoder lacks.
5. **Measured, not assumed** (label-free, §9 L1-L3):
   - codebook perplexity;
   - code gain: decoder error with codes shuffled within the batch versus with the inferred codes, where zero gain
     means the decoder ignores the code;
   - the code-swap grid: if a code changes pixels all over the frame, it is encoding the frame, not a move.

### 3a. Measured before locking G1: the visible-action ceiling and a one-step lag

`eval/visible_ceiling.py` (config `configs/lam_agreement.yaml`), no model involved. It ran on the 35 frozen-val
episodes of the 200k set on the Mac (2,085 decision events, 9,604 eligible steps; binomial SE ≈ 0.01). The full
388-episode split is on the pod volume. At decision events, Pac-Man's real movement stands in for a perfect code, and
the best mapping to the label is fitted on half the episodes and scored on the other half:

| movement feature | → new RAM direction (G1's label) | → true 9-way action |
|---|---|---|
| RAM, one step (i-1 → i) | **0.914** | 0.548 |
| RAM, two steps (i-1 → i+1) | 0.830 | 0.481 |
| RAM one step + previous direction | 0.956 | – |
| pixel detector at 64px, same transition as the action (i-1 → i) | **0.412** | 0.269 |
| pixel detector at 64px, the transition after (i → i+1) | **0.788** | 0.455 |
| majority class | 0.357 | 0.232 |

Over all eligible steps, RAM movement → 9-way action is 0.291 (majority 0.175). That is the ceiling for all-step
9-way agreement.

What this means:
1. **G1 must be scored against the new direction, not the 9-way action.** Diagonal actions (DOWNLEFT and DOWNRIGHT
   are the two most common at turns) cap 9-way agreement at 0.55 for any frame-based code.
2. **The frames lag RAM by about one step.** Over all steps, the 64px pixel displacement correlates with the RAM
   displacement one step *earlier* at 0.76 / 0.81 (x / y), against 0.63 / 0.72 for the same step. So the effect of
   `actions[j]` shows in pixels mainly in `frames[j+1] -> frames[j+2]`. The 4-frame max-pool and the Atari's
   rendering are the likely reasons. A LAM code for transition j therefore mostly reflects `actions[j-1]`.
   Consequences:
   - G1 and G2 compare `codes[i]` with `actions[i-1]` (lag 1).
   - `lam_mapped` replay feeds the keys of `actions[t-1]` one step later, matching the labeled model's own built-in
     latency.
   - Play time needs no change: a key's code acts on the very next transition.
3. **0.75 was not comfortably below the ceiling a pixel model faces, so G1 was set at 0.70.** With lag 1, perfect perception gives 0.91
   (the RAM one-step equivalent), but the detector's centroid, a crude pixel reader, gets only 0.788. A learned encoder
   should read pixels better than the centroid, but that is not guaranteed. The decision is in §12.

## 4. The known risk: Pac-Man's input matters only at junctions (item 6)

The action changes the next frame only where Pac-Man *can* change direction: at a junction (turn), or anywhere by
reversing. Everywhere else he keeps going, and the action is invisible. The four ghosts also make choices at junctions,
so a naive LAM could spend its 3 bits on ghost decisions. Defences, in order of strength:

a. **The context explains ghosts.** With 97 steps of reach, ghost motion between junctions, pen releases and
   frightened phases already in progress are predictable. Only ghost *junction choices* stay surprising.

b. **The bottleneck favours one consistent factor.** 3 bits cannot describe four independent ghost choices. They can
   describe the one sprite that moves on every step. Pac-Man is also the only sprite that reverses in the middle of a
   corridor: the recorder's ε = 0.1 random actions and sticky random segments create frequent, large surprises that
   no ghost produces.

c. **Arm B: player-weighted reconstruction.**
   - Loss weight = 1 + λ·[pixel within r px of Pac-Man's position in frame i-1 or frame i], with λ = 10 and r = 4 px
     at 64px. Frames where the detector misses Pac-Man get uniform weight.
   - Positions come from `eval/detectors.py` over the cache (pixels only). On real 64px frames it found Pac-Man in all
     4,864 frames checked, with ~0.6 px position error.
   - This tells the LAM which sprite is the player, which is a domain prior, not an action label.

d. **Train both arms and pick without labels.**
   - Arms A (generic) and B (player-weighted) are trained identically apart from the loss weight.
   - Among the arms that pass the label-free checks L1-L3, the one with the higher **T_det** is used.
   - T_det: on val transitions where the *detector* sees Pac-Man's movement direction change (a turn visible in
     pixels), the accuracy of that arm's label-free key mapping at predicting the new direction. It is G1 computed
     from pixels instead of RAM and labels.
   - Tie-break: if A is within 0.03 of B, A wins, because it is the generic, Genie-like model.
   - The choice is committed before labels are read. The other arm's label numbers are then reported for information
     and never used to switch arms.

e. **Leakage is measured** (eval only, RAM): the **ghost-turn code-switch lift**. Take steps where Pac-Man goes
   straight and is not at a junction. It is P(code changes | some ghost turned) minus P(code changes | no ghost
   turned). Near 0 means the codes ignore ghosts.

If this risk comes true anyway, G1 catches it: codes that track ghosts agree with Pac-Man's turns at about chance level.

## 5. Staged plan (item 2)

**Stage 0: code and preparation (no training).**
- Implement what Stage 1 needs:
  - `lam.py`, `train_lam.py`, `tools/pac_positions.py`, `tools/lam_codes.py`, `tools/lam_keymap.py`,
    `eval/lam_agreement.py`;
  - `dataset.py`'s `action_source`;
  - the LAM and WM configs.

  The Stage 2-4 code (`eval_rollouts` action modes, `junction_test.py` with `configs/eval_junction.yaml`,
  `server_lam.py`) is written and committed before Stage 2 starts. The junction constants are committed before any
  junction run.
- Checks:
  - the label-firewall test: the random-actions double run in §2;
  - LAM and prior smoke runs on the Mac with the local 200k set;
  - on the pod, `wandb login --verify` and a 150-step smoke run.
- Disk:
  - run `df -h` / `du -sh` first;
  - record the sha256 of `frames128_2m.npy` and `frames128_2m_meta.npz` in the handoff;
  - grep configs, scripts and running processes for any user of them;
  - delete only if nothing uses them.
- Run `tools/pac_positions.py` over `frames64_2m.npy`: 2.2M frames, ~6.6 ms each, ~20 min on 16 vCPU. Output
  `data/cache/pac64_2m.npy`.
- Re-run `eval/visible_ceiling.py` on the full 388-episode val split and record it in the handoff next to the
  35-episode numbers.

**Stage 1: LAM alone. This is the cheap go/no-go.**
1. Train `lam-A` and `lam-B` (30k steps each).
2. Infer codes for all transitions (`tools/lam_codes.py`, which writes `data/cache/lam_codes_{A,B}.npy`).
3. Label-free checks L1-L3, then T_det, then the arm choice. **Commit the choice to the handoff.**
4. Train the prior for the chosen arm, calibrate τ, write the key map (`tools/lam_keymap.py`).
5. Only now run `eval/lam_agreement.py` (labels) for G1 and G2. **If either fails: STOP and report.** No
   retuning to pass, no third arm, no diffusion training. Cost at this stop point: Stage 0 + 1 ≈ $3.25.
6. **Stop for your review regardless of the outcome.** Show the label-free checks, the arm choice, G1 and G2 before
   any Stage 2 spend.

**Stage 2: world models.** Only after Stage 1 passes, you say go, and the RunPod balance is $25 or more. Two pods in
parallel:
- `m1-2M-latent-ctx6s16` (100k at 1e-4), then `-ft-uniform` (15k at 1e-5);
- `m1-2M-noact-ctx6s16`, then `-ft-uniform`, with the same recipe.

W2 is read at **20k steps** on both runs, from the val denoise loss logged at the 20k eval. If latent is not lower,
pause both runs and ask before spending the remaining ~80k + 15k steps. W1 is checked at the end.

**Stage 3: evaluation** of the three models (§8). First validate the junction test on the labeled and no-action
models, then score the latent model against P1-P3.

**Stage 4: side demo.** `serve/server_lam.py` on port 8002: fps check D1, then your playtest. The main demo and
`configs/serve.yaml` are untouched.

## 6. World model on latent codes (item 3)

- **Configs.** Copies of `m1-2M-ctx6s16.yaml` and `m1-2M-ctx6s16-ft-uniform.yaml` with only these changes:
  - `data.action_source: lam`
  - `data.lam_codes: data/cache/lam_codes_<arm>.npy`
  - `model.n_actions: 8`
  - own `train.checkpoint_dir` and `eval.out_dir`

  Same cache, frozen split, context offsets, UNet, EDM settings, context-noise augmentation, batch 64, seed 0, and
  100k @ 1e-4 then 15k @ 1e-5 with a fresh optimizer from the first run's `model1_latest.pt`.
- **What the WM sees.** Each of its 10 action slots holds the code of the transition leaving that context frame,
  exactly as it held the action before. The slot for offset -1 holds the code of the transition being predicted. At
  play time that code is the one the key chose.
- **No-action baseline.** Identical recipe with `data.action_source: none` and `model.n_actions: 1`. Every slot then
  carries the same constant embedding, which is equivalent to no action input with the architecture otherwise unchanged.
- **From scratch, not fine-tuned from ctx6s16**, because those weights learned from true labels.
- **Own folders**:
  - `checkpoints/m1-2M-{latent,noact}-ctx6s16{,-ft-uniform}/`
  - `outputs/...`
  - `eval/results/...`

  The candidate is the annealed `-ft-uniform` EMA checkpoint, as before.
- **In-training evals** use LAM codes throughout: val denoise loss on the fixed val batches, and the 75-step rollout
  GIF replays the codes the LAM gave the real frames. No labels are involved.

## 7. Play time (item 4)

### 7.1 Key → code mapping from unlabeled data
- For each code c, take all train transitions labelled c and measure Pac-Man's displacement from frame j to j+1 with
  the detector. Keep moves of at least 0.3 px.
- **dir(c)** is the majority direction if its share (the purity) is at least 0.5.
- Otherwise c is an **environment code**: deaths, board resets, or anything not tied to Pac-Man's motion. The same
  applies when Pac-Man is undetected in most of c's transitions.
- Key k ∈ {up, right, down, left} gets the set D_k = {c : dir(c) = k}.
- Written to `checkpoints/lam-<arm>/keymap.json` with the full purity table.
- Labels only *verify* it: G2, plus a table of this mapping against the label-optimal one.
- A model-based cross-check comes free in Stage 3: the junction test run with the raw key-to-code mapping (`lam_direct`).

### 7.2 The prior: which codes can happen next here
- `p(code | past)`: a small conv classifier (the LAM encoder trunk, over the 10-frame ctx6s16 context, *without* the
  next frame).
- 20k steps, batch 256, cross-entropy on the chosen arm's codes, final weights, wandb `lam-<arm>-prior`.
- In a horizontal corridor p("up") ≈ 0, because the data never shows Pac-Man moving up there. The prior is a legal-move
  model learned without labels.
- **τ_legal** is the 1st percentile of p(true code) over train transitions. This is label-free: 99% of real
  transitions clear it, so it only blocks moves the data essentially never shows.
- **τ_env = 0.5**, fixed.

### 7.3 Resolver (every frame)
```
π = prior(context);  c_prev = LAM code of the most recent (generated) transition
if  π[environment codes].sum() >= τ_env:        code = argmax over environment codes   # the world forces an event (e.g. death)
elif keys held and max π over D_k (k in keys) >= τ_legal:
                                                code = that argmax                     # requested move is possible; two keys = diagonal
elif π(c_prev) >= τ_legal:                      code = c_prev                          # no-op: keep doing what you were doing
else:                                           code = argmax π                        # forced (corner / wall): do what the data does
```
This mirrors the game.
- A held direction is buffered until it becomes possible, then taken. That is how people pre-turn at junctions.
- A reversal is always possible.
- Releasing all keys keeps Pac-Man going.
- A diagonal is two held keys: the more likely legal one wins, like the Atari diagonals.

### 7.4 What the "no-op" code is
There is **no single no-op code** in Pac-Man, because NOOP in the real game means "keep going": Pac-Man never stops for
lack of input. So the no-op is context-dependent: *the code the LAM gave the last transition, if the prior still allows
it*. If the codes turn out to be motion directions (the expected case), that means "keep moving the same way". If they
turn out to be events ("turn" / "continue"), it means "continue". Either way it is the right semantics, provided the
History holds *re-inferred* codes (§7.5). If the codebook has a "stationary" code (majority displacement below
0.3 px), `argmax π` picks it at walls naturally. Verified with labels as part of G2, on NOOP steps and on steps whose
action matches Pac-Man's current direction.

### 7.5 Re-inference
- After each generated frame, the LAM encoder labels the transition the model actually produced. That code, not the
  commanded one, is stored in the History (`History.relabel_last`).
- In training, the context codes always agree with the context frames. This keeps it that way when the model does not
  obey a command, and it keeps `c_prev` meaningful.
- Cost: one 4-frame encoder pass, ~1 ms.
- Ablation (reported only): store the commanded code instead.

### 7.6 Session start and server
- `new_session` seeds the History with val frames plus their **LAM codes**, never the recorded actions.
- `serve/server_lam.py` subclasses `World` and overrides only `new_session` and `predict`.
- Config: `configs/serve_lam.yaml`. Port 8002, because 8001 is RunPod's nginx.
- The stats line shows the active code and its mapped direction.
- Latency: today ~28.7 ms p50, plus ~2 ms for the prior and the encoder, against a 66.7 ms budget.

## 8. Evaluation (item 5)

**Models.** All three use the same 2.2M data, split, ctx6s16 layout and 100k + 15k anneal recipe.

| role | checkpoint | conditioning |
|---|---|---|
| labeled baseline | `m1-2M-ctx6s16-ft-uniform` (exists; serves the demo) | true actions |
| candidate | `m1-2M-latent-ctx6s16-ft-uniform` | LAM codes |
| no-action control | `m1-2M-noact-ctx6s16-ft-uniform` | nothing |

**Action modes in `eval_rollouts.py` / `junction_test.py`** (`rollout.action_source`):
- `recorded`: the labeled model replays the recorded actions, as today.
- `lam_mapped`: **the primary mode for the latent model.** The recorded action becomes the pressed keys (via
  `metrics.ACTION_DIRS`), then the resolver picks the code. The label enters only as "what the player pressed", which
  is exactly play time. The keys of `actions[t-1]` are applied one step later, because of the measured lag (§3a).
- `lam_oracle`: the code the LAM gives the *real* transition. This is an upper bound: the codes saw the real next
  frame, so any non-action information they carry leaks into the rollout. Reported, never gated.
- `lam_direct` (junction test only): key → most frequent code in D_k, with no resolver. Reported ablation.
- `none`: the no-action model.

**Metric group A, LAM agreement** (Stage 1, `eval/lam_agreement.py`, all 388 val episodes; see G1 and G2 in §9).

Why the gate is at *decision events*: ft-uniform's eval has ~84 direction changes per 450 steps (0.19 per step,
roughly 40k in the val split). On the other ~80% of steps the action has no visible effect. So agreement over all steps
is bounded well below 100% for *any* frame-based code. Reported alongside the gate:
- **9-way agreement over all steps** with its **visible-action ceiling**: the accuracy of the best mapping from Pac-Man's
  RAM movement class (none/U/R/D/L) to the true 9-way action, which is the most any frame-derived code could reach.
- Normalised mutual information between code and action.
- Confusion matrices.
- Agreement on straight steps.
- The ghost-turn lift (§4e).
- **Heading-conditioned accuracy**: the mapping from (code, previous direction) to new direction. This catches codes
  that mean "turn relative to heading" rather than absolute directions.

**Metric group B, junction controllability** (new, `eval/junction_test.py`). This is the counterfactual test that
proves the codes carry control.
- States: 200 val states (seed 0), each with Pac-Man 2-6 steps before a junction of the maze graph (3 or more exits),
  at least 100 steps into the episode, and outside life-loss windows. Each state's context is its real frames.
- Commands: each of the 4 directions held for 16 steps, 3 sampler seeds. That is 2,400 short rollouts per model and
  mode, a few minutes of GPU time.
- **Success** for a legal command (an exit of that junction; reversal is always legal): within 10 steps of Pac-Man
  reaching the junction, the detector's movement direction (`metrics.movement_direction`) equals the command for 3
  consecutive steps.
  - "Reaching the junction" means coming within 1.5 px of its pixel position.
  - Junction pixel positions come from a RAM ↔ detector fit on real frames.
  - The exact constants go into `configs/eval_junction.yaml`, committed before any run.
- **R_j** = (latent − no-action) / (labeled − no-action): the share of the labeled model's control that the codes recover.
- Also reported:
  - **spread**: mean pairwise distance between Pac-Man's positions after 16 steps under the 4 commands; for the
    no-action model this is only sampling noise;
  - illegal commands: how often Pac-Man enters wall pixels, and whether he continues straight.
- Why not rely on replayed responsiveness alone: the PPO agent is predictable from the screen, so a no-action model can
  "respond" to replayed turns by guessing them. In the junction test *we* choose the command, uniformly over legal
  exits, so guessing earns only ~1/(number of exits).

**Metric group C, standard rollouts** (existing harness): the same 10 episodes x 3 seeds, start step 100, 3-step Euler,
ctx sigma 0.
- Wall IoU, pellet IoU, Pac-Man error, responsiveness @15/150/450.
- Ghost count under ORIGINAL and FAIR gating.
- Pen and release metrics (`pen_timer_analysis.py`, `residual_ghosts.py`, `fair_ghosts.py`).
- `compare_runs.py` across all three models and modes.

## 9. Pre-registered pass/fail criteria (item 8)

Identical to the pre-registration section in `notes/handoff.md` (committed 2026-09-29).

**Hard rule:** true actions and RAM are used only for evaluation, never for training or model selection. Same frozen
split `configs/val_episodes_2m.json` (388 val episodes, sha256 `7ba75f89...`) for every model and every measure.

**Measured before locking G1** (`eval/visible_ceiling.py`, `configs/lam_agreement.yaml`, no model involved; the 35
frozen-val episodes of the 200k set, 2,085 decision events). Pac-Man's real movement stands in for a perfect code;
best mapping fitted on half the episodes, scored on the other half:

| feature | → new RAM direction | → true 9-way action |
|---|---|---|
| RAM, one step | 0.914 | 0.548 |
| 64px pixel detector, same transition as the action (lag 0) | 0.407 | 0.269 |
| **64px pixel detector, next transition (lag 1)** | **0.788** | 0.455 |
| majority class | 0.357 | 0.232 |

The frames lag RAM by about one step: pixel displacement correlates with the RAM displacement one step earlier at
0.76 / 0.81 (x / y) vs 0.63 / 0.72 for the same step. **The lag is fixed at 1: G1 and G2 compare `codes[i]` with
`actions[i-1]`, and no other lag is tried later.** Replayed evaluations (`lam_mapped`) apply the keys of `actions[t-1]`
one step later. G1 uses the new direction, not the 9-way action, which diagonal actions cap at 0.55 for any
frame-based code.

**Precondition for Stage 1 (pod):** `eval/visible_ceiling.py` rerun on the full 388-episode split. If
`pix_lag1 -> new_direction` is **below 0.76**: stop and ask. No bar is changed without the owner.

**Stage 1: LAM** (arms A and B only). Label-free checks run first, and the arm choice is committed before any label
is read.

| id | check | reads | pass | on failure |
|---|---|---|---|---|
| L1 | label-free key map (Pac-Man detector displacement per code over the code's own transition): every direction key gets at least one code with purity ≥ 0.5 | pixels | all four | arm excluded |
| L2 | codebook perplexity on val | pixels | ≥ 4.0 (of 8) | arm excluded |
| L3 | code gain: val decoder MSE with codes shuffled within the batch / with inferred codes, paired bootstrap (1,000 resamples) | pixels | > 1, 95% CI excludes 1 | arm excluded |
| sel | T_det: on val transitions where the detector sees Pac-Man's movement direction change, accuracy of the arm's label-free key map at the new direction | pixels | higher wins; A if within 0.03 | no arm left: **STOP and report** (labels never read) |
| **G1** | decision events (RAM direction changes at i to a direction contained in `actions[i-1]`; the first 90 steps, 12 before to 90 after each life loss, and tunnel wraps excluded): accuracy of the best many-to-one map `codes[i]` → new direction, fitted on one half of the val episodes and scored on the other (2-fold, pooled) | labels + RAM | **≥ 0.70 and ≥ majority-class rate + 0.30** | **STOP and report** |
| **G2** | keys held in `actions[i-1]` → resolver on the real context up to frame i → code ĉ. Compare dir(ĉ) with dir(`codes[i]`), where dir() is the label-free key map (UP / RIGHT / DOWN / LEFT, or NONE for environment codes) and agreement means equal labels | labels | **≥ 0.65 at decision events and ≥ 0.85 over all eligible steps** | **STOP and report** |

On any Stage 1 failure: stop and report. No retuning to pass. Any LAM other than arms A and B needs its own
pre-registration. Cost at this stop point: Stage 0 + 1 ≈ $3.25. **After Stage 1, stop for review in any case:**
label-free checks, arm choice, G1 and G2 are shown to the owner before any Stage 2 spend.

**Stage 2: world models** (`m1-2M-latent-ctx6s16` and `m1-2M-noact-ctx6s16`, 100k @ 1e-4, then `-ft-uniform` 15k @
1e-5; from scratch; uniform window sampling, no `events:` section). Two pods, only after Stage 1 passes, the owner says
go, and the RunPod balance is $25 or more.

| id | check | pass | on failure |
|---|---|---|---|
| W1 | both runs reach 100k + 15k with finite losses; parent checkpoints byte-identical after the anneal | yes | debug and rerun, no conclusions |
| W2 | val denoise loss on identical fixed val batches, read at the **20k-step** eval of both runs | latent strictly lower than no-action | **pause both runs and ask** |

W2 alone proves little: the codes were computed from the true next frame, so a lower loss can come from information
about that frame rather than from control. P1 is the real test.

**Stage 3: evaluation.** The latent model is scored in `lam_mapped` mode (true action → keys, one step later →
resolver → code).

| id | check | pass |
|---|---|---|
| V | junction-test validity, checked before the latent model is scored | labeled success **≥ 0.80** and labeled − no-action **≥ 0.30**; otherwise the test is fixed on those two models only |
| **P1** | junction controllability success (200 val states x 4 held directions x 3 seeds; constants in `configs/eval_junction.yaml`, committed before any run). Bootstrap 95% CI over the 200 states (1,000 resamples) for success and for R_j = (latent − none) / (labeled − none) | **success ≥ 0.70 and R_j ≥ 0.70** (point estimates; CIs reported) |
| P2 | responsiveness @15 (standard rollouts) | ≥ 0.70 |
| P3 | world quality @450 | wall IoU ≥ 0.943; pellet IoU ≥ 0.792; Pac-Man error ≤ 23.8 px; longest pen stay median 70-95; 0/30 parked 200+; release hazard lag 91-150 ≥ 0.90; released within 150 steps orange ≥ 95%, cyan ≥ 90%, pink ≥ 80% |

**Stage 4: side demo.** D1: `serve/server_lam.py` (port 8002; resolver, re-inference and watchdog on) streams ≥ 14.9
fps over 450 frames with p95 < 66.7 ms and no false watchdog resets.

**Decision rule:** "learned controls work" only if every Stage 1 gate passes, plus W1, W2, V, P1, P2, P3 and D1. The
main demo stays on `m1-2M-ctx6s16-ft-uniform` either way. The side demo stays up for playtesting only if P1-P3 and D1
pass.

**Reported, not gating:**
- the unchosen arm's numbers;
- 9-way agreement against its ceiling, NMI, heading-conditioned accuracy, ghost-turn code-switch lift;
- `lam_oracle` and `lam_direct` results, and the re-inference ablation;
- every metric for the no-action model;
- ghost counts under both gatings;
- wall entries under illegal commands;
- how often the resolver forces environment codes.

## 10. Compute, cost, disk, time (item 7)

Pod: a secure-cloud RTX 4090 at ~$0.74/h, created with `--min-cuda-version 13.0` and the network volume attached.
Speeds come from the handoff:
- ft-uniform: 15k steps in 30.6 min;
- `bench_128.py`: 64px training at 10.65 it/s;
- ctx6s16: 100k steps in 302 min on a slow host.

| stage | work | pod-hours (wall) | cost |
|---|---|---|---|
| 0 | pod bootstrap and wandb gate, Pac-Man positions (~20 min CPU), smoke runs | ~1.0 | ~$0.75 |
| 1 | 2 LAMs (~55 min each at ~9 it/s), codes (~5 min each), checks and selection (~20 min), prior (~20 min), key map and τ (~10 min), label eval (~20 min) | ~3.3 | ~$2.50 |
| 2 | 2 WMs x (100k ≈ 3.3 h + 15k ≈ 0.5 h) | ~7.6 GPU-h (3.8 h wall on 2 pods, 7.6 h on 1) | ~$5.60 |
| 3 | rollouts: latent mapped + oracle + no-action (~20 min each); junction test, 4 model-modes (~30 min); pen/ghost analyses (~45 min) | ~2.3 | ~$1.70 |
| 4 | side-demo fps check, playtest | ~1.0 | ~$0.75 |
| **total** | | **~15** | **~$11.50 (budget $20)** |

The stop point after Stage 1 costs Stage 0 + 1 ≈ 4.3 pod-hours ≈ **$3.25**.

**Balance.** On 2026-09-29 `runpodctl user` shows **−$0.06**. There are no pods, and the network volume
`v3kyag5rhv` ("pacworld", 100 GB, EU-RO-1) still exists at ~$0.01/h. Nothing on the pod can run until the balance is
topped up. Keep it at **$25 or more before Stage 2**: a zero balance has twice terminated every pod, the demo included.

**Disk.**

The handoff's last reading (2026-09-22) was 81 of 100 GB used. Procedure, on the pod:
1. `df -h /workspace && du -sh /workspace/pacworld/data/cache/* /workspace/pacworld/checkpoints /workspace/pacworld/eval/results`
2. `sha256sum data/cache/frames128_2m.npy data/cache/frames128_2m_meta.npz`, recorded in the handoff.
3. `grep -rn frames128 configs/ tools/ *.py eval/ serve/` and `ps aux`. Configs that point at it are only a problem if
   something runs them.
4. Delete only if nothing uses them.

- **Delete:** `data/cache/frames128_2m.npy` and `frames128_2m_meta.npz` (33.7 GiB). The 128px run failed its
  pre-registration, nothing here uses it, and it rebuilds in ~12 min. Usage drops to about 47 GB.
- **Keep:**
  - `frames64_2m.npy` (27 GB; this project's data);
  - raw episodes (needed for rebuilds and for RAM in eval);
  - all checkpoints (~0.4 GB each);
  - the pellet recordings.
- **New data (~5-6 GB total):**
  - Pac-Man positions ~10 MB;
  - codes 2 x 2.2 MB;
  - LAM checkpoints ~0.5 GB, prior ~30 MB;
  - 4 WM checkpoint folders ~1.5 GB;
  - outputs ~0.2 GB;
  - eval predictions ~3 GB. Junction-test frames are kept for one seed only; positions are kept for all.

**Calendar time.**
- Stage 0: one working session to implement, with your review of each piece.
- Stage 1: about half a day, including a pause for your read of G1 and G2.
- Stage 2: ~4 h on two pods (fine overnight).
- Stage 3: about half a day.
- Stage 4: an hour plus your playtest.
- Roughly 3 days end to end.

## 11. Files and folders

**Written in Stage 0 (2026-09-29):**
- `lam.py`: encoder, EMA vector quantiser, decoder around `model1.UNet.run`, prior, player weight, `resolve()`.
- `train_lam.py --part lam|prior`: wandb logging, code-swap grids, `--firewall-scramble-seed` / `--dump` for the
  firewall test.
- `tools/pac_positions.py` (pixel detector over the cache), `tools/lam_codes.py` (codes array),
  `tools/lam_keymap.py` (label-free map and τ calibration), `tools/lam_checks.py` (L1-L3, T_det, `--select`).
- `eval/visible_ceiling.py` (committed with the pre-registration) and `eval/lam_agreement.py` (G1, G2, reported
  numbers). Both read `configs/lam_agreement.yaml`, which holds the pre-registered bars.
- `tools/firewall_test.py` (one double run), `tools/firewall_suite_mac.sh` (all entry points plus a labeled
  negative control that must fail), `tools/stage0_pod.sh` (the pod half of Stage 0; deletes nothing).
- Configs:
  - `configs/lam-{A,B}.yaml`, and `lam-{A,B}-mac.yaml` for the Mac smoke;
  - `configs/m1-2M-{latent,noact}-ctx6s16{,-ft-uniform}.yaml`. The latent configs point at
    `lam_codes_ARM.npy` until the arm is chosen;
  - `configs/m1-{noact,latent,labels}-mac.yaml` for smoke and firewall tests.
- `README.md`: the latent-action section, including that arm B uses the pixel detector.

**Edited, additive only:**
- `model1.py`: `UNet.forward` split into the conditioning step and `run()` (outputs verified bit-identical).
- `dataset.py`: `apply_action_source` (the firewall; labels deleted for `lam` / `none`) and `History.relabel_last`.
- `train_model1.py`: refuses `events:` without labels; `--firewall-scramble-seed` / `--dump`; codes named in the
  rollout GIF.

**Still to write before Stage 2 / 3:**
- `eval/junction_test.py` and `configs/eval_junction.yaml`, committed before any junction run;
- the `rollout.action_source` modes in `eval/eval_rollouts.py`;
- `eval_m1-2M-{latent,noact}-ctx6s16-ft-uniform.yaml`;
- `serve/server_lam.py` and `configs/serve_lam.yaml`.

**Not touched:** `serve/server.py`, `configs/serve.yaml`, the running demo, every existing checkpoint.

Every script takes `--seed`, and every training run (both LAMs, the prior, all four WM runs) logs to wandb project
`pacworld`.

**Failure modes and what each would look like:**

| failure | symptom |
|---|---|
| codes track ghosts | G1 near chance, high ghost-turn lift, code-swap grid changes ghosts |
| the decoder ignores the code | L3 ratio ≈ 1, low perplexity |
| one-step turns too small to see at 64px | G1 moderate but straight-step agreement high |
| codes relative to heading | G1 low but heading-conditioned accuracy high. This would need a new pre-registration with a heading-aware key map. |
| the WM ignores codes because they are mostly predictable from context | W2 or P1 fails; the no-action model shows it |
| codes carry more than the action | `lam_oracle` scores well above `lam_mapped` |

## 12. Decisions

**Made (2026-09-29):**
1. Run both arms, A and B. The README must say that B uses the pixel detector.
2. The resolver is primary; direct key-to-code mapping is an ablation.
3. 8 codes.
4. Two pods for Stage 2, but only after Stage 1 passes and the balance is $25 or more.
5. Delete the 128px cache after the checks in §10.

**Settled 2026-09-29 (after §3a):**
- G1: lag fixed at 1 (`codes[i]` against `actions[i-1]`; no other lag is tried later). Bar 0.70 and at least majority + 0.30.
- G2 compares directions, not exact codes. Bar 0.65 at decision events and 0.85 over all steps.
- Replayed evals delay keys by one step.
- Before Stage 1, the ceiling is rerun on the full 388-episode split. If the lag-1 pixel ceiling is below 0.76: stop
  and ask. No bar is changed without the owner.
