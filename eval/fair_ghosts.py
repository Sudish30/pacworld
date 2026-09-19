"""Fair ghost accounting with a frightened-blue class, frightened-phase durations, respawn causes, disappearances.

Inference-free. Reads the saved rollouts of one evaluated checkpoint:
  --raw    ghost_diag/raw.npz (coloured-ghost detections per condition)      default <out_dir>/ghost_diag/raw.npz
  --preds  the generated frames of the SAME rollouts (eval_rollouts.py --save-all-preds) for --condition
           default <out_dir>/model1/preds_all.npy
and the ground-truth frames and RAM of the evaluated episodes. Frightened ghosts are found with
detectors.MazeReference.frightened_blobs in both worlds; a blob's weight is converted to a ghost count with the
median weight of a ground-truth frightened blob.

  1. headline ghost count, two gatings side by side (the original one is never replaced):
       original   eligible steps from ground-truth RAM, coloured ghosts only (what summary.csv reports)
       fair       also drops the reset period around the MODEL'S OWN respawns, and counts blue ghosts as present
     and the split of missing coloured ghosts into reset / frightened / penned / truly gone
  2. frightened-phase durations, model vs real game (RAM timer), and what is left of the ghosts afterwards
  3. in-maze disappearances (detected outside the pen, then no coloured detection for --loss-gap steps),
     tagged: own life loss / turned blue / next to Pac-Man / tunnel edge / over another ghost
  4. the model's own respawns: was a ghost touching Pac-Man just before the death animation?

  python eval/fair_ghosts.py --config configs/eval_m1-2M-ctx6s16.yaml --seed 0
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from common import load_config  # noqa: E402
from dataset import episode_path  # noqa: E402
import detectors as D  # noqa: E402
from eval_rollouts import write_csv  # noqa: E402
from pen_timer_analysis import OCC_PIXELS, OCC_RESID, PEN_BOX, in_pen  # noqa: E402
from residual_ghosts import own_respawns, pct  # noqa: E402

_REF = None
MAXB = 4


def _init(cfg):
    global _REF
    _REF = D.load_reference(cfg)


def _frames_pass(frames):
    """Per frame: Pac-Man position, frightened blobs (row, col, weight; nan-padded), pen occupied."""
    n = len(frames)
    pac, blobs, occ = np.full((n, 2), np.nan), np.full((n, MAXB, 3), np.nan), np.zeros(n, bool)
    bgbox = _REF.bg64[PEN_BOX]
    for t, f in enumerate(frames):
        f = np.asarray(f)
        pres = _REF.pellet_presence(f)
        s = _REF.sprites(f, pres)
        if s["pac"] is not None:
            pac[t] = s["pac"][:2]
        for j, b in enumerate(sorted(_REF.frightened_blobs(f, pres), key=lambda b: -b[2])[:MAXB]):
            blobs[t, j] = b
        occ[t] = (np.linalg.norm(f[PEN_BOX].astype(float) - bgbox, axis=-1) > OCC_RESID).sum() >= OCC_PIXELS
    return pac, blobs, occ


def runs_of(b, gap=0):
    """(start, length) of True runs in b, bridging gaps of up to `gap` False steps."""
    out, i, n = [], 0, len(b)
    while i < n:
        if b[i]:
            j = last = i
            while j < n and (b[j] or j - last <= gap):
                if b[j]:
                    last = j
                j += 1
            out.append((i, last - i + 1))
            i = last + 1
        else:
            i += 1
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--condition", default="autoregressive, 3 steps")
    p.add_argument("--raw", help="raw.npz with the coloured detections of --condition")
    p.add_argument("--preds", help="preds_all.npy of the same rollouts")
    p.add_argument("--tag", default="", help="suffix of the output csv names")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--loss-gap", type=int, default=15)
    p.add_argument("--touch-px", type=float, default=4.5, help="Pac-Man and a ghost closer than this count as touching")
    p.add_argument("--blue-gap", type=int, default=12, help="blue-free steps bridged inside one frightened phase (ghosts blink near the end)")
    a = p.parse_args()
    cfg = load_config(a.config)
    gate = cfg["gating"]
    out = ROOT / cfg["out_dir"] / "ghost_diag"
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(a.raw or out / "raw.npz", allow_pickle=True)
    ci = list(z["conditions"]).index(a.condition)
    start, H, R, G = int(z["start"]), int(z["horizon"]), len(z["episode_seed"]), D.GHOST_NAMES
    det, pos, pac = z[f"c{ci}_det"], z[f"c{ci}_pos"], z[f"c{ci}_pac"]
    gt_det, gt_pos, E = z["gt_det"], z["gt_pos"], z["eligible"]
    preds = np.load(a.preds or ROOT / cfg["out_dir"] / "model1" / "preds_all.npy", mmap_mode="r")
    seeds = [int(s) for s in z["episode_seed"]]
    eps = {s: np.load(episode_path(cfg["data"], s, ROOT)) for s in sorted(set(seeds))}
    rams = {s: e["ram"] for s, e in eps.items()}

    with Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
        ref = D.load_reference(cfg)
        check = np.array([ref.ghost_mass(np.asarray(preds[0, k]))["total"] for k in range(0, H, 50)])
        assert np.allclose(check, z[f"c{ci}_mass"][0, 0:H:50]), "--preds are not the rollouts of --condition in --raw"
        m = pool.map(_frames_pass, [preds[r] for r in range(R)])     # the full saved rollout (longer than the scored horizon)
        gt_frames = {s: D.downsample_frames(e["frames"][start:start + H]) for s, e in eps.items()}
        g_by_seed = dict(zip(gt_frames, pool.map(_frames_pass, list(gt_frames.values()))))
    m_blobs_full = np.stack([x[1] for x in m])                      # (R, saved horizon, MAXB, 3): frightened phases are timed on this
    m_pac, m_blobs, m_occ = (np.stack([x[i][:H] for x in m]) for i in range(3))
    g_pac, g_blobs, g_occ = (np.stack([g_by_seed[s][i] for s in seeds]) for i in range(3))

    # ---- ground-truth state
    fr = np.stack([rams[s][start:start + H, gate["frightened_ram"]] > 0 for s in seeds])
    pen = np.stack([np.stack([in_pen(rams[s], g)[start:start + H] for g in G], -1) for s in seeds])
    gt_ev = []
    for s in seeds:
        losses = np.nonzero(np.diff(rams[s][:, gate["lives_ram"]].astype(int)) < 0)[0] + 1 - start
        gt_ev.append([int(k) for k in losses if 0 <= k < H])
    m_ev = [own_respawns(pac[r], H) for r in range(R)]
    inside = gt_pos[gt_det & pen]
    lo, hi = np.percentile(inside, 1, axis=0) - 1.0, np.percentile(inside, 99, axis=0) + 1.0
    zone = lambda P: (P[..., 0] >= lo[0]) & (P[..., 0] <= hi[0]) & (P[..., 1] >= lo[1]) & (P[..., 1] <= hi[1])
    mz = zone(pos) & det

    # ---- frightened class: calibration on ground truth, then a ghost count per frame
    gw = g_blobs[..., 2]
    w_ref = float(np.nanmedian(gw[fr]))
    n_blobs = lambda B: (~np.isnan(B[..., 2])).sum(-1)

    def blue_count(B, coloured):
        est = np.nansum(np.maximum(1, np.round(B[..., 2] / w_ref)) * ~np.isnan(B[..., 2]), axis=-1)
        return np.minimum(est, 4 - coloured).astype(int)

    g_col, m_col = gt_det.sum(-1), det.sum(-1)
    g_blue, m_blue = blue_count(g_blobs, g_col), blue_count(m_blobs, m_col)
    print(f"rollouts {R}, horizon {H}, condition '{a.condition}'")
    print(f"frightened class, ground-truth check: frames with a blue blob while RAM says frightened {100 * (n_blobs(g_blobs)[fr] > 0).mean():.0f}%, "
          f"while RAM says not frightened {100 * (n_blobs(g_blobs)[~fr] > 0).mean():.2f}% (false positives); single-ghost blob weight {w_ref:.1f}")
    print(f"  ground truth, frightened frames: coloured {g_col[fr].mean():.2f} + blue {g_blue[fr].mean():.2f} = {(g_col + g_blue)[fr].mean():.2f} ghosts on screen "
          f"(eaten ghosts are off screen until they leave the pen again)")
    rows = []

    # ---- 1. headline, both gatings
    reset = np.zeros((R, H), bool)
    for r in range(R):
        for k in m_ev[r]:
            reset[r, max(0, k - gate["death_anim_steps"]):k + gate["start_of_life_steps"]] = True
    fair = E & ~reset
    m_present = np.minimum(m_col + m_blue, 4)

    def window_mean(v, mask, a0, b0):
        per = [v[r, a0:b0][mask[r, a0:b0]].mean() for r in range(R) if mask[r, a0:b0].any()]
        return (float(np.mean(per)), len(per)) if per else (float("nan"), 0)

    w = cfg["rollout"]["window"]
    print(f"\n=== 1. ghost count on screen (ground truth = 4 when nothing is eaten or penned) ===")
    print(f"{'window':>22s} | {'ORIGINAL gating, coloured only':^44s} | {'FAIR gating (own respawns dropped), blue = present':^52s}")
    print(f"{'':>22s} | {'model':>10s}{'ground truth':>14s}{'rollouts':>10s}{'':10s} | {'model':>10s}{'of which blue':>15s}{'ground truth':>14s}{'rollouts':>10s}")
    for name, a0, b0 in ([(f"last {w} steps to {h}", h - w, h) for h in cfg["rollout"]["gated_horizons"] if h <= H] +
                         [("steps 131-250", 130, 250), (f"steps 251-{H}", 250, H)]):
        o_m, o_n = window_mean(m_col, E, a0, b0)
        o_g, _ = window_mean(g_col, E, a0, b0)
        f_m, f_n = window_mean(m_present, fair, a0, b0)
        f_b, _ = window_mean(m_blue, fair, a0, b0)
        f_g, _ = window_mean(g_col, fair, a0, b0)
        print(f"{name:>22s} | {o_m:>10.2f}{o_g:>14.2f}{o_n:>10d}{'':10s} | {f_m:>10.2f}{f_b:>15.2f}{f_g:>14.2f}{f_n:>10d}")
        rows.append({"part": "1_headline", "key": name, "original_model": o_m, "original_gt": o_g, "original_n": o_n,
                     "fair_model": f_m, "fair_model_blue": f_b, "fair_gt": f_g, "fair_n": f_n})
    print(f"  fair gating removes {100 * (1 - fair.sum() / max(E.sum(), 1)):.0f}% of the originally eligible steps "
          f"({gate['death_anim_steps']} steps before to {gate['start_of_life_steps']} steps after each of the model's {sum(map(len, m_ev))} own respawns)")

    print(f"\nmissing coloured ghosts on ORIGINALLY eligible steps, split (ghosts per step):")
    print(f"{'window':>16s}{'missing':>9s}{'own reset period':>18s}{'frightened (blue)':>19s}{'penned':>9s}{'truly gone':>12s}{'ground truth missing':>22s}")
    for name, a0, b0 in (("steps 131-250", 130, 250), (f"steps 251-{H}", 250, H)):
        sel = E[:, a0:b0]
        n = max(sel.sum(), 1)
        miss = (4 - m_col)[:, a0:b0] * sel
        in_reset = reset[:, a0:b0]
        blue = np.minimum(m_blue[:, a0:b0], 4 - m_col[:, a0:b0]) * sel * ~in_reset
        rest = miss * ~in_reset - blue
        penned = rest * m_occ[:, a0:b0]
        gone = rest * ~m_occ[:, a0:b0]
        vals = [miss.sum() / n, (miss * in_reset).sum() / n, blue.sum() / n, penned.sum() / n, gone.sum() / n, ((4 - g_col)[:, a0:b0] * sel).sum() / n]
        print(f"{name:>16s}{vals[0]:>9.2f}{vals[1]:>18.2f}{vals[2]:>19.2f}{vals[3]:>9.2f}{vals[4]:>12.2f}{vals[5]:>22.2f}")
        rows.append({"part": "1_missing_split", "key": name, "original_model": vals[0], "original_gt": vals[5], "original_n": int(sel.sum()),
                     "fair_model": vals[1], "fair_model_blue": vals[2], "fair_gt": vals[3], "fair_n": vals[4]})

    # ---- 2. frightened phases
    print(f"\n=== 2. frightened-phase durations (steps; context reach of m1-2M-ctx6s16 is 97) ===")
    real = []
    for s, ram in rams.items():                                   # whole episodes, exact RAM timer
        real += [L for s0, L in runs_of(ram[:, gate["frightened_ram"]] > 0) if s0 + L < len(ram)]
    g_pix = [(r, s0, L) for r in range(R) for s0, L in runs_of(n_blobs(g_blobs)[r] > 0, gap=a.blue_gap)]
    m_pix = [(r, s0, L) for r in range(R) for s0, L in runs_of(n_blobs(m_blobs)[r] > 0, gap=a.blue_gap)]
    # ground truth is the same episode for the 3 seeds of a rollout: keep one copy per episode
    seen, g_unique = set(), []
    for r, s0, L in g_pix:
        if (seeds[r], s0) not in seen:
            seen.add((seeds[r], s0))
            g_unique.append((r, s0, L))

    def describe(label, L, cens):
        L = np.array(L)
        print(f"{label:46s} n={len(L):3d}  median {pct(L, 50):5.0f}  p10 {pct(L, 10):5.0f}  p25 {pct(L, 25):5.0f}  p75 {pct(L, 75):5.0f}  p90 {pct(L, 90):5.0f}  max {L.max() if len(L) else 0:4d}  cut by horizon {cens}")
        return {"part": "2_frightened", "key": label, "original_model": pct(L, 50), "original_gt": pct(L, 25), "original_n": len(L),
                "fair_model": pct(L, 75), "fair_model_blue": pct(L, 90), "fair_gt": float(L.max()) if len(L) else float("nan"), "fair_n": cens}

    rows.append(describe("real game, RAM timer (whole episodes)", real, 0))
    rows.append(describe("real game, blue pixels (rollout window)", [L for r, s0, L in g_unique if s0 + L < H], sum(s0 + L >= H for r, s0, L in g_unique)))
    rows.append(describe("model, blue pixels", [L for r, s0, L in m_pix if s0 + L < H], sum(s0 + L >= H for r, s0, L in m_pix)))
    ram_in_window = [L for s in sorted(set(seeds)) for s0, L in runs_of(rams[s][start:start + H, gate["frightened_ram"]] > 0) if s0 + L < H]
    print(f"  pixel-vs-RAM check on the real game inside the window: RAM phases {sorted(ram_in_window)}, pixel phases {sorted(L for r, s0, L in g_unique if s0 + L < H)}")
    print("  model phases (rollout, first step, length, * = still blue at the horizon): " +
          ", ".join(f"({r},{s0 + 1},{L}{'*' if s0 + L >= H else ''})" for r, s0, L in m_pix))
    cens = [L for r, s0, L in m_pix if s0 + L >= H]
    if cens:
        print(f"  phases still blue at the horizon: {len(cens)}, already lasting {sorted(cens)} steps (real phases end after {min(real) if real else 0}-{max(real) if real else 0})")
    HF = m_blobs_full.shape[1]
    m_full = [(r, s0, L) for r in range(R) for s0, L in runs_of(n_blobs(m_blobs_full)[r] > 0, gap=a.blue_gap) if L >= 5]
    print(f"  model phases over the full {HF}-step rollouts (5+ steps; * = still blue at step {HF}): " +
          ", ".join(f"({r},{s0 + 1},{L}{'*' if s0 + L >= HF else ''})" for r, s0, L in m_full))
    ended = np.array([L for r, s0, L in m_full if s0 + L < HF])
    open_ = np.array([L for r, s0, L in m_full if s0 + L >= HF])
    rows.append(describe(f"model, blue pixels, full {HF}-step rollouts, ended", ended, len(open_)))
    if len(ended):
        print(f"  ended model phases by length: <60: {(ended < 60).sum()}, 60-89: {((ended >= 60) & (ended < 90)).sum()}, 90-104 (context reach 97): {((ended >= 90) & (ended < 105)).sum()}, "
              f"105-140 (real game {min(real)}-{max(real)}): {((ended >= 105) & (ended <= 140)).sum()}, 141+: {(ended > 140).sum()}")
    if len(open_):
        print(f"  never-ending phases (still blue at step {HF}): {len(open_)}, lasting {sorted(open_.tolist())} steps so far")
    done = [(r, s0, L) for r, s0, L in m_pix if s0 + L < H]
    if done:
        Ls = np.array([L for _, _, L in done])
        print(f"  model phases by length: <30: {(Ls < 30).sum()}, 30-59: {((Ls >= 30) & (Ls < 60)).sum()}, 60-89: {((Ls >= 60) & (Ls < 90)).sum()}, "
              f"90-104 (at the context reach): {((Ls >= 90) & (Ls < 105)).sum()}, 105+: {(Ls >= 105).sum()}")
        before = np.array([m_col[r, max(0, s0 - 10):max(1, s0 - 2)].max() for r, s0, L in done])
        after = np.array([m_col[r, min(H - 1, s0 + L + 5):min(H, s0 + L + 40)].max() if s0 + L + 5 < H else -1 for r, s0, L in done])
        ok = after >= 0
        print(f"  coloured ghosts before a phase {before[ok].mean():.2f} -> within 40 steps after it {after[ok].mean():.2f} "
              f"(ghosts that come back coloured: {100 * (after[ok] >= before[ok]).mean():.0f}% of phases keep or regain their count)")
    gb = [(r, s0, L) for r, s0, L in g_unique if s0 + L < H]
    if gb:
        gb_before = np.array([g_col[r, max(0, s0 - 10):max(1, s0 - 2)].max() for r, s0, L in gb])
        gb_after = np.array([g_col[r, min(H - 1, s0 + L + 5):min(H, s0 + L + 40)].max() if s0 + L + 5 < H else -1 for r, s0, L in gb])
        okg = gb_after >= 0
        print(f"  real game, same measure: before {gb_before[okg].mean():.2f} -> after {gb_after[okg].mean():.2f}")

    # ---- 3. in-maze disappearances, blue-aware
    evs = []
    for r in range(R):
        for gi, g in enumerate(G):
            k = 0
            while k + 1 + a.loss_gap <= H:
                if det[r, k, gi] and not mz[r, k, gi] and not det[r, k + 1:k + 1 + a.loss_gap, gi].any():
                    nxt = np.nonzero(det[r, k + 1:, gi])[0]
                    back = int(nxt[0]) + 1 if len(nxt) else -1
                    here = pos[r, k, gi]
                    bl = m_blobs[r, k + 1:k + 1 + a.loss_gap, :, :2]
                    turned_blue = bool(np.nanmin(np.hypot(bl[..., 0] - here[0], bl[..., 1] - here[1]), initial=np.inf) < 8) if (~np.isnan(bl[..., 0])).any() else False
                    others = [np.hypot(*(pos[r, k, gj] - here)) for gj in range(4) if gj != gi and det[r, k, gj]]
                    d_pac = np.hypot(*(pac[r, k] - here)) if not np.isnan(pac[r, k]).any() else np.nan
                    cause = ("own life loss" if any(-10 <= e - k <= 40 for e in m_ev[r]) else "turned blue" if turned_blue else
                             "next to Pac-Man" if d_pac < 6 else "tunnel edge" if (here[1] < 5 or here[1] > 59) else
                             "over another ghost" if (others and min(others) < 5) else "open maze")
                    evs.append({"rollout": r, "ghost": g, "step": k + 1, "pos0": float(here[0]), "pos1": float(here[1]), "cause": cause,
                                "absent_steps": back if back > 0 else H - k, "permanent": back < 0, "returns_in_pen": bool(back > 0 and mz[r, k + back, gi])})
                    k += a.loss_gap
                k += 1
    print(f"\n=== 3. in-maze disappearances (coloured ghost outside the pen, then undetected {a.loss_gap}+ steps): {len(evs)} ===")
    print(f"{'cause':22s}{'events':>8s}{'permanent':>11s}{'absent ghost-steps':>20s}{'ghosts/step':>13s}")
    for cause in ("own life loss", "turned blue", "next to Pac-Man", "tunnel edge", "over another ghost", "open maze"):
        sub = [e for e in evs if e["cause"] == cause]
        steps = sum(e["absent_steps"] for e in sub)
        print(f"{cause:22s}{len(sub):>8d}{sum(e['permanent'] for e in sub):>11d}{steps:>20d}{steps / (R * H):>13.3f}")
        rows.append({"part": "3_disappearances", "key": cause, "original_model": len(sub), "original_gt": sum(e["permanent"] for e in sub), "original_n": steps,
                     "fair_model": steps / (R * H), "fair_model_blue": "", "fair_gt": "", "fair_n": ""})
    drift = [e for e in evs if e["cause"] in ("next to Pac-Man", "tunnel edge", "over another ghost", "open maze")]
    print(f"drift candidates (not a life loss, not frightened): {len(drift)} events, {sum(e['permanent'] for e in drift)} permanent, "
          f"{sum(e['absent_steps'] for e in drift) / (R * H):.3f} ghosts/step")
    rows.append({"part": "3_disappearances", "key": "drift candidates", "original_model": len(drift), "original_gt": sum(e["permanent"] for e in drift),
                 "original_n": sum(e["absent_steps"] for e in drift), "fair_model": sum(e["absent_steps"] for e in drift) / (R * H), "fair_model_blue": "", "fair_gt": "", "fair_n": ""})
    if evs:
        write_csv(out / f"fair_disappearances{a.tag}.csv", evs)

    # ---- 4. what precedes a respawn: a ghost touching Pac-Man before the death animation?
    def touch(events, P_pac, P_g, d_g, B):
        res = []
        for r in range(R):
            for k in events[r]:
                a0, b0 = max(0, k - gate["death_anim_steps"] - 25), max(1, k - gate["death_anim_steps"] + 4)
                dmin, dblue = np.inf, np.inf
                for t in range(a0, b0):
                    if np.isnan(P_pac[r, t]).any():
                        continue
                    for gi in range(4):
                        if d_g[r, t, gi]:
                            dmin = min(dmin, np.hypot(*(P_g[r, t, gi] - P_pac[r, t])))
                    for j in range(MAXB):
                        if not np.isnan(B[r, t, j, 0]):
                            dblue = min(dblue, np.hypot(*(B[r, t, j, :2] - P_pac[r, t])))
                res.append((r, k, dmin, dblue, k < gate["death_anim_steps"] + 25))
        return res

    print(f"\n=== 4. respawns: was a coloured ghost within {a.touch_px} px of Pac-Man in the 25 steps before the death animation? ===")
    for world, res in (("real game", touch(gt_ev, g_pac, gt_pos, gt_det, g_blobs)), ("model (own respawns)", touch(m_ev, m_pac, pos, det, m_blobs))):
        if world == "real game":                                  # one copy per episode
            uniq, s_ = [], set()
            for r, k, d1, d2, early in res:
                if (seeds[r], k) not in s_:
                    s_.add((seeds[r], k))
                    uniq.append((r, k, d1, d2, early))
            res = uniq
        res = [x for x in res if not x[4]]
        n = max(len(res), 1)
        hit = sum(d1 < a.touch_px for _, _, d1, _, _ in res)
        blue_only = sum(d1 >= a.touch_px and d2 < a.touch_px for _, _, d1, d2, _ in res)
        none = [x for x in res if x[2] >= a.touch_px and x[3] >= a.touch_px]
        print(f"{world:22s} n={len(res):3d}  ghost touching {100 * hit / n:4.0f}%   only a BLUE ghost touching {100 * blue_only / n:4.0f}%   nothing near {100 * len(none) / n:4.0f}%"
              f"   (nearest coloured ghost when nothing touches: median {pct([x[2] for x in none if np.isfinite(x[2])], 50):.1f} px; no ghost on screen at all: {sum(not np.isfinite(x[2]) for x in none)})")
        rows.append({"part": "4_respawn_cause", "key": world, "original_model": hit / n, "original_gt": blue_only / n, "original_n": len(res),
                     "fair_model": len(none) / n, "fair_model_blue": "", "fair_gt": "", "fair_n": ""})
    per_roll = np.array([len(e) for e in m_ev])
    gaps = [b - a_ for e in m_ev for a_, b in zip(e, e[1:])]
    print(f"model respawns per rollout: mean {per_roll.mean():.2f} (real {np.mean([len(e) for e in gt_ev]):.2f}); "
          f"gaps between consecutive own respawns: median {pct(gaps, 50):.0f} steps, under 100 steps: {sum(g_ < 100 for g_ in gaps)} of {len(gaps)}")
    write_csv(out / f"fair_ghosts{a.tag}.csv", rows)
    print(f"\nwrote {out}/fair_ghosts{a.tag}.csv")


if __name__ == "__main__":
    main()
