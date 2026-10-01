# LAM v2: PRE-REGISTRATION (2026-10-01; reviewed and approved by the owner before this commit; no v2 GPU run exists)

v1 is final and reported as a negative result: both arms collapsed, L1-L3 failed, labels were never read (handoff,
2026-09-30). v2 changes **only** the five items below. Everything else is v1 unchanged:
- the label firewall and its double-run test;
- the encoder (frames j-2..j+1, lag 1), the 8-code codebook and the 30k-step recipe;
- the pixel-detector player weight (lambda 10, r 4 px);
- the prior (on ctx6s16), key map and resolver;
- L1-L3, G1 (≥ 0.70 and majority + 0.30, lag 1) and G2 (by direction, 0.65 / 0.85), with the same data, split and
  bars.

Config: `configs/lam-v2.yaml`. Code: `lam.py`, `train_lam.py` (`--stop-at`), `tools/lam_pilot.py`.

## 1. The five changes

**(i) Decoder context: the last frame only (offset [-1]).** (Owner's change to the draft, which had 4 frames.)
- Root cause of v1: the PPO agent's moves are predictable from the screen. Four frames still show Pac-Man's heading,
  so the decoder could predict the next frame without the code.
- From a single frame the heading is ambiguous. The code is then needed on almost every step, and should come to
  mean Pac-Man's direction of motion.
- Caveat: frames are max-pooled over the 4 skipped emulator frames. A single frame may therefore show the *axis* of
  motion as a smear, but not its sign.
- `[-1]` is the last of ctx6s16's offsets, so the same `gather_context` call is used and sliced.
- The prior keeps ctx6s16, because it has to judge legality in the world model's own context at play time.

**(ii) Collapse-resistant quantiser: VQ plus a code-usage entropy term (chosen over FSQ), weight tied to the loss
scale.**
- The loss adds w_ent x T, with T = E_z[H(p(k|z))] − H(E_z[p(k|z)]), the entropy objective of MAGVIT-v2's LFQ (Yu
  et al. 2024).
- p(k|z) = softmax(−||ẑ − ê_k||² / τ), with ẑ and ê the L2-normalised encoder output and code, τ = 0.1. Gradient
  reaches the encoder only; the codebook keeps its EMA updates.
- **The weight** (owner's decision): the term's magnitude at initialisation equals the reconstruction loss.

      w_ent = L_recon(batch 1) / |T(batch 1)|

  - L_recon is the player-weighted reconstruction loss and T the unweighted entropy term, both on the first training
    batch (seed 0), before any update.
  - It is computed once, then fixed for the whole run and saved in the checkpoint (`vq.entropy_weight`), so the
    resumed run after the pilot uses the same value.
  - **Value.** The registered run's first batch comes from the 2.2M-frame set, which exists only on the pod volume.
    So the exact value is fixed by this formula, seed 0 and the config, and is printed at step 1 and recorded in the
    handoff with the pilot. On the Mac's 200k set (batch 16, seed 0) the same formula gives
    **w_ent = 0.0833** (L_recon 0.015876, T −0.190663). With batch 64 on the pod, T will be closer to its
    large-batch value; the expected weight is of the same order.
- Why not FSQ: in v1 the encoder output itself became constant (commitment 0.00000, and dead-code restarts
  re-seeded from identical outputs). FSQ has no codebook to collapse, but it rounds a constant output to a single
  code just as well. The entropy term penalises exactly that state.
- Its risk: it can force spread-out codes that mean nothing. The pilot's code-gain condition catches that.

**(iii) Code conditioning does not start at zero influence.**
- The decoder's AdaGN scale/shift projections start at N(0, 0.02) instead of zeros.
- The decoder's output convolution starts at N(0, 0.001) instead of zeros. Without this, the zero output conv blocks
  all code influence at initialisation (found by the first $0 Mac check).
- The world model is untouched, and v1's configs behave as they ran (v1 checkpoints load unchanged).

**(iv) One arm, player-weighted (arm B style). No arm A.**
- The T_det arm selection falls away; L1-L3 still have to pass.

**(v) The code is also broadcast as extra input channels to the decoder's first conv.** (Approved by the owner as a
fifth change.)
- The 32-dim quantised code vector is repeated over the 64x64 image and concatenated to the input frame: the first
  conv has 3 + 32 input channels, default initialisation.
- The AdaGN path of v1 stays. This gives the code a second, direct path whose influence is large from step 0.
- Reason: in the first Mac checks, spread-out codes were not used by the decoder within 300 steps under either
  entropy weight.

## 2. Pilot first (the owner's rule; no second pilot)

- Train 2,500 steps (`train_lam.py --stop-at 2500`; the cosine schedule spans the full 30k from the start).
- Score with `tools/lam_pilot.py`. Continue to the full 30k-step run **only if all three hold at step 2,500**:
  1. **perplexity ≥ 4**;
  2. **code gain (shuffled / inferred val MSE) > 1.01**;
  3. **Pac-Man share > ghost share in the attribution split.**
- 1 and 2 are measured on the same fixed val batches as in v1.
- **The attribution split**, label-free, pixels only:
  - 1,000 fixed val contexts (seed 0). Each is decoded with all 8 codes.
  - Spread = the per-pixel variance of the 8 outputs, averaged over RGB.
  - A pixel is Pac-Man's if it lies within 4 px of his detected centre in the last context frame. It is a ghost's if
    within 4 px of any detected ghost (the four colours and every frightened blob). Positions come from the pixel
    detector.
  - Pixels near both go to a separate "both" bucket and count for neither.
  - Share = spread summed over the bucket and all 1,000 contexts, divided by the total spread. Contexts where a
    sprite is not detected stay in the denominator.
  - The comparison is of raw shares. Up to four ghosts cover about four times Pac-Man's area, which works against
    Pac-Man; the per-pixel means are reported too, but do not decide.
- Otherwise: stop and report. No second pilot.
- The full run **continues from the pilot checkpoint** (`--resume`; same seed, schedule and entropy weight), so the
  pilot's 2,500 steps are the first 2,500 of the registered 30k.

## 3. Pre-spend check

**(a) Most likely ways it fails.**
1. **Ghost capture: the main risk.** A single frame makes the ghosts' headings ambiguous too, and it cannot time pen
   releases or frightened phases. Four ghosts' moves are then as surprising as Pac-Man's, and a 3-bit code could
   spend itself on them.
   - The player weight (x11 within 4 px of Pac-Man) works against this, but four ghosts can outweigh one sprite.
   - Pilot condition 3 guards it.
2. The decoder still ignores the code. The max-pooled smear and the maze (corridors allow two directions, and the
   agent rarely reverses) may tell it enough. Code gain then stays near 1.00 and the pilot stops the run.
3. The entropy term forces diversity that means nothing: perplexity ≥ 4 with gain ≤ 1.01, and the pilot stops.

**(b) Cheapest tests, in order.**
1. **$0, Mac, before any GPU** (section 3b): code influence at initialisation is non-zero; a 300-step CPU run does
   not collapse. Mechanics only, not a gate.
2. **The GPU pilot (2,500 steps)**, with the three conditions above.
3. At the end, as before: L1 (every direction gets a code), T_det, and, if the gates run, the ghost-turn code-switch
   lift (eval-only, RAM).

**(c) Differences from the published method (Genie's LAM, Bruce et al. 2024) and the risk each creates.**

| difference | Genie | v2 | risk |
|---|---|---|---|
| encoder | ST-ViViT transformer over all previous frames + next frame | CNN over 4 frames (j-2..j+1) | the encoder cannot use long history. Low risk for a move that is visible over 1-2 frames. |
| decoder context | transformer over all previous frames | conv UNet over **1 frame** | **ghost capture, the main risk.** Ghost headings are as ambiguous as Pac-Man's, and pen and frightened timing are invisible, so the code may describe ghosts. Guarded by pilot condition 3 and the player weight. |
| code path | conditioning embeddings | AdaGN conditioning + broadcast input channels | none known; a stronger path than v1's |
| quantiser | VQ, 8 codes, no entropy term reported | VQ, 8 codes, EMA, dead-code restart + MAGVIT-v2 entropy term | codes may be spread without meaning (caught by the gain condition). The weight is set by a fixed formula, not tuned. |
| conditioning init | not reported | AdaGN and output conv non-zero at init | slightly less stable early training, low risk |
| loss | pixel reconstruction | pixel MSE x (1 + 10 near Pac-Man), weights from the pixel detector | domain knowledge Genie does not use (already declared in the README) |
| data | ~30k hours of 2D platformer video, many games | 2.2M frames of one game from one PPO agent (+ ε-random) | the agent's actions are predictable from the screen. This is the root of v1's failure, and the reason for the 1-frame decoder. |
| frames | single video frames | max-pooled over 4 emulator frames | one frame can show the axis of motion as a smear (not its sign), which weakens change (i) |
| action-effect timing | not an issue at their frame rate | the action shows mainly one transition later (lag 1) | the code's own transition carries less of the action. Unchanged from v1; ceiling 0.78 pixels at lag 1. |

## 3b. $0 Mac mechanics checks (local 200k set, batch 16, CPU; mechanics only, not results and not a gate)

**On the final config (2026-10-01), `configs/lam-v2-mac.yaml`:**
1. **Code influence at initialisation is non-zero.** The spread of the decoder output across the 8 codes (std, mean
   over pixels) is 2.5e-3 on a 16-context val batch (draft: 9.1e-4). The decoder still starts close to "copy the
   last frame" (|output − last frame| = 1.0e-2). The decoder's first conv has 35 input channels; 23.18M parameters.
2. **Tied entropy weight.** First batch: L_recon 0.015876, |T| 0.190663, so w_ent = 0.0833.
   - The term matches the reconstruction loss only at step 1, as specified. By step 300 the reconstruction loss has
     fallen to about 0.008 while the weighted term is −0.136, so the term is then about 16x the reconstruction loss.
3. **300-step run** (batch 16, eval every 100 steps on 20 fixed val batches):

   | step | perplexity | code gain | val MSE |
   |---|---|---|---|
   | 100 | 1.05 | x1.000 | 0.00423 |
   | 200 | 6.61 | x1.005 | 0.00379 |
   | 300 | 6.07 | x1.013 | 0.00327 |

   - No collapse after step 100; 0 code restarts.
   - The decoder starts to use the codes: this is the first Mac run with a code gain above 1.000.
4. **The pilot tool runs end to end** (`tools/lam_pilot.py --mechanics` on the step-300 checkpoint, 1,000 val
   contexts; Pac-Man detected in 1,000, a ghost in 946):
   - spread share: Pac-Man 0.110, ghosts 0.137, both 0.005, elsewhere 0.748;
   - mean spread per pixel: Pac-Man 4.0e-4, ghosts 1.7e-4, elsewhere 3.3e-5.
   - At step 300 on the Mac, condition 3 would **not** hold (0.110 < 0.137), although per pixel the codes act 2.4
     times more strongly near Pac-Man than near ghosts. Most of the spread is still elsewhere in the frame.
   - This is a 300-step CPU run at batch 16 on the 200k set. It is not a gate and predicts little about step 2,500
     at batch 64, but it shows that ghost capture is a live risk and that condition 3 can bind.

**On the earlier draft (2026-09-30; 4-frame decoder, no input-channel path), kept for the record:**
- Code influence at init: 0.00 with the AdaGN change only (the zero output conv blocked it); 9.1e-4 with the output
  conv change.
- Entropy weight 0.1: perplexity 7.20 at step 200 and 6.21 at step 300; code gain x1.000; the entropy term was about
  25x the reconstruction loss.
- Entropy weight 0.01: perplexity 1.00 at steps 100 and 200, 5.28 at step 300; code gain x1.000.
- These led to the tied weight and to change (v).

## 4. Criteria

Unchanged from v1 (pre-registration 6bb6039, with this file's three-condition pilot rule added):
- L1-L3 on the single arm;
- G1 ≥ 0.70 and ≥ majority + 0.30 at lag 1;
- G2 by direction, ≥ 0.65 at decision events and ≥ 0.85 over all steps.

On any failure: stop and report; no retuning. A v3 would need its own pre-registration.

## 5. Cost

- Mac checks: $0.
- Pilot: 2,500 steps. The full run adds 27,500 steps, then codes, checks, prior, key map and gates.
- On the current RTX 2000 Ada ($0.24/h) the LAM's speed is not yet measured. On the RTX PRO 4500 v1 ran at 8.4 it/s
  ($0.72/h). The 1-frame decoder has 35 instead of 30 input channels, so the cost per step is about the same.
- Estimate: under 6 GPU-hours on either card, **about $2-5**.
