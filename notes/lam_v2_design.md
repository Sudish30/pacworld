# LAM v2: design for review (DRAFT, not committed as a pre-registration yet)

v1 is final and reported as a negative result: both arms collapsed, L1-L3 failed, labels were never read (handoff,
2026-09-30). v2 changes **only** the four items the owner listed. Everything else is v1 unchanged:
- the label firewall and its double-run test;
- the encoder (frames j-2..j+1, lag 1), the 8-code codebook and the 30k-step recipe;
- the pixel-detector player weight (lambda 10, r 4 px);
- the prior, key map and resolver;
- L1-L3, G1 (≥ 0.70 and majority + 0.30, lag 1) and G2 (by direction, 0.65 / 0.85), with the same data, split and
  bars.

## 1. The four changes

**(i) Decoder context: the last 4 consecutive frames.**
- The decoder sees offsets [-4, -3, -2, -1] instead of ctx6s16's 10 strided frames.
- These are the last four of ctx6s16's offsets, so they are gathered by the same `gather_context` call and simply
  sliced.
- The prior keeps ctx6s16, because it has to judge legality in the world model's own context at play time. That is
  not a v2 change: it was the v1 design and is untouched.
- Why this change: in v1 the 97-step decoder predicted the next frame well from context alone (val MSE 0.00023), so
  3 bits of code bought nothing.

**(ii) Collapse-resistant quantiser: VQ plus a code-usage entropy term (chosen over FSQ).**
- The loss adds lambda_ent x (E_z[H(p(k|z))] − H(E_z[p(k|z)])), the entropy objective of MAGVIT-v2's LFQ (Yu et al.
  2024).
- p(k|z) = softmax(−||ẑ − ê_k||² / τ), with ẑ and ê the L2-normalised encoder output and code. τ = 0.1, lambda_ent =
  0.1 (MAGVIT-v2's weight). Gradient reaches the encoder only; the codebook keeps its EMA updates.
- The first term makes each assignment confident; the second spreads usage over the batch.
- Why not FSQ: in v1 the encoder output itself became constant (commitment 0.00000, and dead-code restarts
  re-seeded from identical outputs). FSQ has no codebook to collapse, but it rounds a constant output to a single
  code just as well, so it would not have prevented what actually happened. The entropy term penalises exactly that
  state.
- Its risk: it can force spread-out codes that mean nothing. The pilot's code-gain condition catches that, because
  meaningless codes do not lower the decoder's error.

**(iii) Code conditioning does not start at zero influence.**
- The decoder's AdaGN scale/shift projections start at N(0, 0.02) instead of zeros (v1 and the world model use zero
  init, as in DiT's adaLN-Zero).
- **And the decoder's output convolution starts at N(0, 0.001) instead of zeros.**
  - Found by the $0 Mac check (section 3b): with only the AdaGN change, the spread of the output across 8 codes at
    initialisation was still exactly 0.00, because the UNet's zero-initialised output conv blocks everything upstream.
  - With both changes the spread is 9.1e-4 at step 0, and the decoder still starts close to "copy the last frame"
    (|output − last frame| = 1.2e-2).
  - v1's config is unaffected (spread 0.00, as it ran).
- So the code changes the decoder output from step 0, and the encoder gets decoder gradient from step 0 rather than
  only the commitment loss.
- The world model is untouched.

**(iv) One arm, player-weighted (arm B style). No arm A.**
- The T_det arm selection falls away; L1-L3 still have to pass.

## 2. Pilot first (the owner's rule; no second pilot)

- Train 2,500 steps. Continue to the full 30k-step run **only if perplexity ≥ 4 AND code gain (shuffled / inferred
  val MSE) > 1.01 at step 2,500**. Both are measured on the same fixed val batches as in v1.
- Otherwise: stop and report.
- The full run then **continues from the pilot checkpoint** (same seed, same schedule), so the pilot's 2,500 steps
  are the first 2,500 of the registered 30k. The cosine schedule is computed for 30k from the start.

## 3. Pre-spend check

**(a) Most likely ways it fails.**
1. The decoder still ignores the code. Even with 4 frames, Pac-Man mostly keeps going, so his next position is
   predictable from recent motion. Code gain then stays at about 1.00 and the pilot stops the run.
2. The entropy term forces diversity that means nothing: perplexity ≥ 4 with gain ≤ 1.01, and the pilot stops.
3. **Ghost capture**, the owner's question. With 4 frames the decoder can no longer predict ghost pen releases or
   frightened-phase timing, which is exactly what the 97-step context was for. It also predicts ghost junction choices
   less well. Those become the largest surprises, and a 3-bit code could spend itself on them instead of on Pac-Man.
   - The player weight (x11 within 4 px of Pac-Man) works against this, but four ghosts can outweigh one sprite.
   - A pen release moves one ghost several pixels at once, a large MSE event.

**(b) Cheapest tests, in order.**
1. **$0, Mac, before any GPU:**
   - at initialisation two different codes give different decoder outputs (the non-zero influence works);
   - a 300-step CPU run on the local 200k set does not collapse (perplexity stays above 1).

   These check the mechanics only; they are not results.
2. **The GPU pilot (2,500 steps)**, with the owner's rule above.
3. **Ghost-capture check at the pilot**, label-free, pixels only: the **attribution split**.
   - Take 1,000 fixed val contexts. Decode each with all 8 codes and compute the per-pixel spread of the 8 outputs.
   - Report the share of that spread within 4 px of Pac-Man, within 4 px of any ghost, and elsewhere. Positions come
     from the pixel detector on the last context frame.
   - Codes that serve Pac-Man put most of their effect near him; codes captured by ghosts put it near ghosts.
   - **Proposal for the owner:** add "Pac-Man share > ghost share" as a third pilot condition. As drafted it is
     reported only, because the owner fixed the pilot rule.
4. At the end, as before: L1 (every direction gets a code), T_det, and, if the gates run, the ghost-turn code-switch
   lift (eval-only, RAM).

**(c) Differences from the published method (Genie's LAM, Bruce et al. 2024) and the risk each creates.**

| difference | Genie | v2 | risk |
|---|---|---|---|
| encoder | ST-ViViT transformer over all previous frames + next frame | CNN over 4 frames (j-2..j+1) | the encoder cannot use long history. Low risk for a move that is visible over 1-2 frames. |
| decoder | transformer over all previous frames | conv UNet over 4 frames | a short decoder context leaves more non-player surprise (ghosts) for the code: risk 3 above |
| quantiser | VQ, 8 codes, no entropy term reported | VQ, 8 codes, EMA, dead-code restart + MAGVIT-v2 entropy term | codes may be spread without meaning (caught by the gain condition). The entropy weight is an extra hyperparameter, fixed in advance. |
| conditioning init | not reported | AdaGN non-zero init | slightly less stable early training, low risk |
| loss | pixel reconstruction | pixel MSE x (1 + 10 near Pac-Man), weights from the pixel detector | domain knowledge Genie does not use (already declared in the README) |
| data | ~30k hours of 2D platformer video, many games | 2.2M frames of one game from one PPO agent (+ ε-random) | the agent's actions are predictable from the screen, so the decoder can guess the move without the code. This is the root of v1's failure. |
| action-effect timing | not an issue at their frame rate | the action shows mainly one transition later (lag 1) | the code's own transition carries less of the action. Unchanged from v1, where it was measured; ceiling 0.78 pixels at lag 1. |

## 3b. Results of the $0 Mac mechanics checks (2026-09-30; local 200k set, batch 16, CPU; mechanics only, not results)

1. **Code influence at initialisation.**
   - v1 config: output spread across 8 codes = 0.00.
   - v2 with the AdaGN change only: also **0.00**. The UNet's zero-initialised output conv blocks all code influence,
     which is why the output-conv init was added to change (iii).
   - v2 with both: 9.1e-4.
2. **300-step run, v2 as drafted (entropy weight 0.1):**
   - perplexity 1.29 at step 100, **7.20** at step 200, **6.21** at step 300. The entropy term prevents the collapse.
   - **Code gain x1.000 at every eval.** The spread-out codes are not yet used by the decoder.
   - **The entropy term is about 25x the reconstruction loss** (−0.134 vs about 0.005). MAGVIT-v2's 0.1 was set for a
     very different loss scale, so at this weight the encoder is mostly pushed to spread codes rather than make them
     informative. That is failure mode (a2).
3. **300-step run with entropy weight 0.01:**
   - collapsed at steps 100 and 200 (perplexity 1.00), then **5.28** at step 300;
   - **code gain x1.000 at every eval**;
   - entropy term −0.011 against reconstruction 0.005.
4. **Reading.**
   - The entropy term does its job (with weight 0.1 the codes are spread from step 200).
   - Neither weight makes the decoder *use* the codes within 300 small CPU steps. At batch 16 that is weak evidence,
     but it points to failure mode (a1)/(a2) as the likely pilot outcome.
   - Whether the gain appears by 2,500 GPU steps at batch 64 is exactly what the pilot tests.
   - If the owner wants to lower the risk before spending, the evidence favours a stronger code path over a different
     entropy weight: for example the code broadcast as extra input channels to the decoder's first conv, so its
     influence starts large. But that would be a fifth change, which the owner's list does not include.

**Decision for the owner before the commit:** the entropy weight. It is inside the approved change (ii), but its value
matters and cannot be retried after the pilot (no second pilot). Options:
- keep 0.1 (MAGVIT-v2's value);
- use 0.01;
- tie it to the loss scale: set it so the entropy term's magnitude at initialisation equals the reconstruction loss,
  computed once from the first batch and then fixed.

## 4. Criteria

Unchanged from v1 (pre-registration 6bb6039, with this file's pilot rule added):
- L1-L3 on the single arm;
- G1 ≥ 0.70 and ≥ majority + 0.30 at lag 1;
- G2 by direction, ≥ 0.65 at decision events and ≥ 0.85 over all steps.

On any failure: stop and report; no retuning. A v3 would need its own pre-registration.

## 5. Cost

- Mac checks: $0.
- Pilot: 2,500 steps. The full run adds 27,500 steps, then codes, checks, prior, key map and gates.
- On the current RTX 2000 Ada ($0.24/h) the LAM's speed is not yet measured. On the RTX PRO 4500 v1 ran at 8.4 it/s
  ($0.72/h). The 4-frame decoder has 12 instead of 30 input channels, so it is slightly cheaper per step.
- Estimate: under 6 GPU-hours on either card, **about $2-5**.
