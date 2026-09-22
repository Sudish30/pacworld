"""What is the residual ghost loss made of? Inference-free: reads the saved rollouts of one evaluated checkpoint.

Inputs (written by eval_rollouts.py --save-all-preds and ghost_diagnostics.py for the config's out_dir):
  <out_dir>/ghost_diag/raw.npz     per-rollout detections and positions, model and ground truth
  <out_dir>/model1/preds_all.npy   the generated frames (for class-agnostic pen occupancy)
  ground-truth RAM of the evaluated episodes

Both worlds are measured with the same pixel criterion: a detected ghost is "in the pen zone" when its detected
position falls inside the box that ground-truth detections occupy while RAM says the ghost is in the pen
(calibrated below, agreement printed). Events are life losses: RAM lives for ground truth, Pac-Man jumping to the
spawn point for the model's own world.

  (a) first-release lag after a life loss, per ghost, model vs ground truth
  (b) re-pen frequency: maze -> pen-zone transitions per 1000 in-maze ghost-steps, at a life loss vs elsewhere
  (c) ghosts that ground truth never pens: when / where / which colour the model loses them, and whether the
      loss coincides with the model's own respawn (a legitimate reset in the model's world)
  (d) decomposition of all missing ghost-steps at steps 251-450, and the ghost count re-gated on the model's
      own respawns (the standard gating only knows the ground-truth life losses)
  (e) every in-maze disappearance (any ghost, in the world's own terms): where, next to what, for how long,
      permanent or not - model vs the same criterion on ground truth

  python eval/residual_ghosts.py --config configs/eval_m1-2M-ctx6s16.yaml --seed 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from common import load_config  # noqa: E402
from dataset import episode_path  # noqa: E402
import detectors as D  # noqa: E402
from eval_rollouts import write_csv  # noqa: E402
from pen_timer_analysis import geom, in_pen  # noqa: E402


def own_respawns(pac, H, size=64):
    """Steps at which the model's Pac-Man jumps to the spawn point (same rule as pen_timer_analysis.py)."""
    GM = geom(size)
    ev = []
    for k in range(1, H):
        a, b = pac[k - 1], pac[k]
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        if np.hypot(*(b - a)) > GM["jump"] and np.hypot(b[0] - GM["spawn"][0], b[1] - GM["spawn"][1]) < GM["spawn_r"]:
            ev.append(k)
    return ev


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--condition", default="autoregressive, 3 steps")
    p.add_argument("--lag-window", type=int, default=150, help="steps after a life loss in which a release is looked for")
    p.add_argument("--loss-gap", type=int, default=15, help="a ghost counts as lost when undetected for this many steps")
    p.add_argument("--respawn-window", type=int, default=90, help="steps after a life loss treated as the reset period (= gating.start_of_life_steps)")
    a = p.parse_args()
    cfg = load_config(a.config)
    out = ROOT / cfg["out_dir"] / "ghost_diag"
    z = np.load(out / "raw.npz", allow_pickle=True)
    ci = list(z["conditions"]).index(a.condition)
    start, H, R, G = int(z["start"]), int(z["horizon"]), len(z["episode_seed"]), D.GHOST_NAMES
    det, pos, pac = z[f"c{ci}_det"], z[f"c{ci}_pos"], z[f"c{ci}_pac"]
    gt_det, gt_pos, E = z["gt_det"], z["gt_pos"], z["eligible"]
    rams = {int(s): np.load(episode_path(cfg["data"], int(s), ROOT))["ram"] for s in np.unique(z["episode_seed"])}

    # ---- ground-truth state from RAM
    pen = np.zeros((R, H, 4), bool)
    valid = np.zeros((R, H), bool)
    gt_ev = []
    for r in range(R):
        ram = rams[int(z["episode_seed"][r])]
        seg = ram[start:start + H]
        losses = np.nonzero(np.diff(ram[:, cfg["gating"]["lives_ram"]].astype(int)) < 0)[0] + 1
        v = seg[:, cfg["gating"]["frightened_ram"]] == 0
        for t in losses:
            k = t - start
            v[max(0, k - cfg["gating"]["death_anim_steps"]):max(0, k + 1)] = False
        valid[r] = v
        gt_ev.append([int(t - start) for t in losses if 0 <= t - start < H])
        for gi, g in enumerate(G):
            pen[r, :, gi] = in_pen(ram, g)[start:start + H]
    been = np.maximum.accumulate(pen, axis=1)
    SIZE = cfg["data"]["size"]
    m_ev = [own_respawns(pac[r], H, SIZE) for r in range(R)]

    # ---- pixel pen zone, calibrated on ground truth: where detections sit while RAM says "in pen"
    inside = gt_pos[gt_det & pen]
    lo, hi = np.percentile(inside, 1, axis=0) - 1.0, np.percentile(inside, 99, axis=0) + 1.0

    def in_zone(P):
        return (P[..., 0] >= lo[0]) & (P[..., 0] <= hi[0]) & (P[..., 1] >= lo[1]) & (P[..., 1] <= hi[1])

    gz, mz = in_zone(gt_pos) & gt_det, in_zone(pos) & det
    print(f"rollouts {R}, horizon {H}, condition '{a.condition}'")
    print(f"pen zone (pixels, from {len(inside)} ground-truth detections in the pen): axis0 {lo[0]:.1f}-{hi[0]:.1f}, axis1 {lo[1]:.1f}-{hi[1]:.1f}")
    print(f"  zone vs RAM on ground-truth detections: in-pen detections inside the zone {100 * gz[gt_det & pen].mean():.1f}%, "
          f"in-maze detections inside the zone {100 * gz[gt_det & ~pen].mean():.1f}%")
    print(f"life losses: ground truth {sum(map(len, gt_ev))}, model's own respawns {sum(map(len, m_ev))}")
    rows = []

    # ---- (a) first-release lag after a life loss
    def release_lags(events, d, zone):
        lags = {g: [] for g in G}
        for r in range(R):
            for i, k in enumerate(events[r]):
                end = min(H, k + a.lag_window, events[r][i + 1] if i + 1 < len(events[r]) else H)
                for gi, g in enumerate(G):
                    out_maze = np.nonzero(d[r, k + 1:end, gi] & ~zone[r, k + 1:end, gi])[0]
                    lags[g].append((int(out_maze[0]) + 1, False) if len(out_maze) else (end - k, end - k < a.lag_window))
        return lags   # (lag, censored-before-the-window-ended)

    print(f"\n=== (a) first step a ghost is seen in the maze after a life loss (window {a.lag_window} steps) ===")
    print(f"{'ghost':8s}{'world':>14s}{'n':>5s}{'released':>10s}{'median':>8s}{'p25':>6s}{'p75':>6s}{'p90':>6s}")
    for g in G:
        for world, lags in (("ground truth", release_lags(gt_ev, gt_det, gz)), ("model", release_lags(m_ev, det, mz))):
            full = [(l, c) for l, c in lags[g] if not c]                       # drop events cut short by the horizon / next loss
            rel = np.array([l for l, c in full if l < a.lag_window])
            frac = len(rel) / max(len(full), 1)
            print(f"{g:8s}{world:>14s}{len(full):>5d}{100 * frac:>9.0f}%{pct(rel, 50):>8.0f}{pct(rel, 25):>6.0f}{pct(rel, 75):>6.0f}{pct(rel, 90):>6.0f}")
            rows.append({"part": "a_release_lag", "ghost": g, "world": world, "key": "", "n": len(full), "value": frac,
                         "median": pct(rel, 50), "p25": pct(rel, 25), "p75": pct(rel, 75), "p90": pct(rel, 90)})

    # ---- (b) re-pen frequency: last in-maze detection followed (within 5 steps) by a detection inside the pen zone
    def repens(d, zone, events):
        at_loss = other = 0
        for r in range(R):
            for gi in range(4):
                t_det = np.nonzero(d[r, :, gi])[0]
                for i in range(len(t_det) - 1):
                    t0, t1 = t_det[i], t_det[i + 1]
                    if not zone[r, t0, gi] and zone[r, t1, gi] and t1 - t0 <= 5:
                        if any(-5 <= t1 - k <= 40 for k in events[r]):
                            at_loss += 1
                        else:
                            other += 1
        maze_steps = int((d & ~zone).sum())
        return at_loss, other, maze_steps

    print(f"\n=== (b) re-pen events: maze -> pen-zone transitions of a detected ghost ===")
    print(f"{'world':14s}{'in-maze ghost-steps':>21s}{'at a life loss':>16s}{'elsewhere':>11s}{'elsewhere / 1000 steps':>24s}")
    for world, d, zone, ev in (("ground truth", gt_det, gz, gt_ev), ("model", det, mz, m_ev)):
        al, ot, ms = repens(d, zone, ev)
        print(f"{world:14s}{ms:>21d}{al:>16d}{ot:>11d}{1000 * ot / max(ms, 1):>24.2f}")
        rows.append({"part": "b_repen", "ghost": "all", "world": world, "key": "elsewhere_per_1000", "n": ms, "value": 1000 * ot / max(ms, 1),
                     "median": al, "p25": ot, "p75": "", "p90": ""})

    # ---- class-agnostic pen occupancy of the generated frames
    ref = D.load_reference(cfg)
    preds = np.load(ROOT / cfg["out_dir"] / "model1" / "preds_all.npy", mmap_mode="r")
    GM = geom(ref.size)
    bgbox = ref.bg64[GM["pen_box"]]
    m_occ = np.array([[(np.linalg.norm(np.asarray(preds[r, k])[GM["pen_box"]].astype(float) - bgbox, axis=-1) > GM["occ_resid"]).sum() >= GM["occ_pixels"]
                       for k in range(H)] for r in range(R)])
    after_own = np.zeros((R, H), bool)          # within the reset period after one of the model's own respawns
    for r in range(R):
        for k in m_ev[r]:
            after_own[r, k:k + a.respawn_window] = True

    # ---- (c) ghosts that ground truth never pens
    never = ~pen & ~been & valid[:, :, None]
    late = np.zeros((R, H), bool)
    late[:, 130:] = True
    print(f"\n=== (c) ghosts ground truth never pens (in the maze since the rollout began), steps 131-{H} ===")
    print(f"{'ghost':8s}{'ghost-steps':>12s}{'detected':>10s}{'missing':>9s} | of the missing: {'after own respawn':>18s}{'pen occupied':>14s}{'gone, pen empty':>17s}")
    for gi, g in enumerate(G):
        m = never[:, :, gi] & late
        n = int(m.sum())
        if not n:
            continue
        miss = m & ~det[:, :, gi]
        nm = int(miss.sum())
        c_res = int((miss & after_own).sum())
        c_pen = int((miss & ~after_own & m_occ).sum())
        c_gone = nm - c_res - c_pen
        print(f"{g:8s}{n:>12d}{100 * (1 - nm / n):>9.0f}%{nm:>9d} | {'':16s}{100 * c_res / max(nm, 1):>17.0f}%{100 * c_pen / max(nm, 1):>13.0f}%{100 * c_gone / max(nm, 1):>16.0f}%")
        for key, v in (("detected", 1 - nm / n), ("missing_after_own_respawn", c_res / max(nm, 1)),
                       ("missing_pen_occupied", c_pen / max(nm, 1)), ("missing_gone", c_gone / max(nm, 1))):
            rows.append({"part": "c_never_penned", "ghost": g, "world": "model", "key": key, "n": n, "value": v, "median": "", "p25": "", "p75": "", "p90": ""})

    # loss events of never-penned ghosts: last detection before >= loss_gap undetected steps
    events = []
    for r in range(R):
        for gi, g in enumerate(G):
            k = 0
            while k < H - 1:
                if det[r, k, gi] and not det[r, k + 1:k + 1 + a.loss_gap, gi].any() and k + 1 + a.loss_gap <= H and never[r, k, gi]:
                    nxt = np.nonzero(det[r, k + 1:, gi])[0]
                    back = int(nxt[0]) + 1 if len(nxt) else -1
                    near = min([k - e for e in m_ev[r]], key=abs, default=None)          # >0: loss happens after the respawn
                    where = ("pen zone" if mz[r, k, gi] else
                             "pen surroundings" if (abs(pos[r, k, gi, 0] - (lo[0] + hi[0]) / 2) < 12 and abs(pos[r, k, gi, 1] - (lo[1] + hi[1]) / 2) < 12)
                             else "maze")
                    events.append({"rollout": r, "ghost": g, "step": k + 1, "where": where, "pos0": float(pos[r, k, gi, 0]), "pos1": float(pos[r, k, gi, 1]),
                                   "steps_from_own_respawn": near if near is not None else "", "at_own_respawn": near is not None and -40 <= near <= 10,
                                   "returns_after": back, "returns_in_pen": bool(back > 0 and mz[r, k + back, gi]),
                                   "gt_still_in_maze_15_later": bool(gt_det[r, min(k + a.loss_gap, H - 1), gi])})
                    k += a.loss_gap
                k += 1
    print(f"\nloss events (detected, then undetected for {a.loss_gap}+ steps, while ground truth keeps the ghost in the maze): {len(events)}")
    if events:
        def share(f):
            return 100 * sum(1 for e in events if f(e)) / len(events)
        print(f"  by ghost:  " + ", ".join(f"{g} {sum(e['ghost'] == g for e in events)}" for g in G))
        print(f"  by place:  " + ", ".join(f"{w} {share(lambda e: e['where'] == w):.0f}%" for w in ("maze", "pen surroundings", "pen zone")))
        print(f"  when:      steps 1-130 {share(lambda e: e['step'] <= 130):.0f}%, 131-250 {share(lambda e: 130 < e['step'] <= 250):.0f}%, 251-{H} {share(lambda e: e['step'] > 250):.0f}%")
        print(f"  coincides with the model's own respawn (loss within [-10, +40] steps of it): {share(lambda e: e['at_own_respawn']):.0f}%")
        print(f"  afterwards: returns inside the pen zone {share(lambda e: e['returns_in_pen']):.0f}%, returns in the maze "
              f"{share(lambda e: e['returns_after'] > 0 and not e['returns_in_pen']):.0f}%, never seen again {share(lambda e: e['returns_after'] < 0):.0f}%")
        ret = [e["returns_after"] for e in events if e["returns_after"] > 0]
        print(f"  steps until seen again (when it returns): median {pct(ret, 50):.0f}, p90 {pct(ret, 90):.0f}")
        fade = [e for e in events if not e["at_own_respawn"]]
        print(f"  NOT at an own respawn ({len(fade)} events): " + ", ".join(f"{g} {sum(e['ghost'] == g for e in fade)}" for g in G) +
              f"; place: " + ", ".join(f"{w} {sum(e['where'] == w for e in fade)}" for w in ("maze", "pen surroundings", "pen zone")) +
              f"; never seen again {sum(e['returns_after'] < 0 for e in fade)}")
        write_csv(out / "residual_loss_events.csv", events)

    # ---- (d) what the missing ghost-steps at 251-H are made of, and the count re-gated on the model's own respawns
    print(f"\n=== (d) all four ghosts, eligible steps 251-{H} (ground-truth gating) ===")
    w = np.zeros((R, H), bool)
    w[:, 250:] = True
    base = E & w
    n_steps = int(base.sum())
    gcount, gt_count = det.sum(-1), gt_det.sum(-1)
    print(f"ghost count: ground truth {gt_count[base].mean():.2f}, model {gcount[base].mean():.2f}  ({n_steps} eligible rollout-steps)")
    fair = base & ~after_own
    print(f"re-gated on the model's own respawns too (drop {a.respawn_window} steps after each): model {gcount[fair].mean():.2f} "
          f"on {int(fair.sum())} steps ({100 * (1 - fair.sum() / max(n_steps, 1)):.0f}% of steps removed); ground truth on the same steps {gt_count[fair].mean():.2f}")
    missing = (4 - gcount) * base
    tot = missing.sum()
    parts = {"in the reset period after the model's own respawn": (missing * after_own).sum(),
             "pen occupied (ghosts parked or legitimately inside; stacked ghosts are not detectable)": (missing * (~after_own & m_occ)).sum(),
             "pen empty: ghost is gone from the frame": (missing * (~after_own & ~m_occ)).sum()}
    print(f"missing ghost-steps: {int(tot)} = {tot / max(n_steps, 1):.2f} ghosts per step")
    for name, v in parts.items():
        print(f"  {100 * v / max(tot, 1):5.1f}%  ({v / max(n_steps, 1):.2f} ghosts/step)  {name}")
        rows.append({"part": "d_missing_split", "ghost": "all", "world": "model", "key": name, "n": int(tot), "value": v / max(tot, 1), "median": v / max(n_steps, 1), "p25": "", "p75": "", "p90": ""})
    rows.append({"part": "d_count", "ghost": "all", "world": "model", "key": "ghost_count_251plus", "n": n_steps, "value": gcount[base].mean(), "median": "", "p25": "", "p75": "", "p90": ""})
    rows.append({"part": "d_count", "ghost": "all", "world": "model", "key": "ghost_count_251plus_regated", "n": int(fair.sum()), "value": gcount[fair].mean(), "median": "", "p25": "", "p75": "", "p90": ""})
    rows.append({"part": "d_count", "ghost": "all", "world": "ground truth", "key": "ghost_count_251plus", "n": n_steps, "value": gt_count[base].mean(), "median": "", "p25": "", "p75": "", "p90": ""})
    per_ghost = [(g, 100 * det[:, :, gi][fair].mean(), 100 * gt_det[:, :, gi][fair].mean()) for gi, g in enumerate(G)]
    print("per-ghost detection on the re-gated steps (model / ground truth): " + ", ".join(f"{g} {m:.0f}% / {t:.0f}%" for g, m, t in per_ghost))
    gt_missing = ((4 - gt_count) * base).sum()
    print(f"for scale, ground truth itself misses {gt_missing / max(n_steps, 1):.2f} ghosts per step on these steps (ghosts inside the pen are hard to detect)")
    # ---- (e) in-maze disappearances in each world's own terms
    def disappearances(d, P, zone, pacpos, events_, ok):
        evs = []
        for r in range(R):
            for gi, g in enumerate(G):
                k = 0
                while k + 1 + a.loss_gap <= H:
                    if d[r, k, gi] and not zone[r, k, gi] and ok[r, k] and not d[r, k + 1:k + 1 + a.loss_gap, gi].any():
                        nxt = np.nonzero(d[r, k + 1:, gi])[0]
                        back = int(nxt[0]) + 1 if len(nxt) else -1
                        others = [np.hypot(*(P[r, k, gj] - P[r, k, gi])) for gj in range(4) if gj != gi and d[r, k, gj]]
                        d_pac = np.hypot(*(pacpos[r, k] - P[r, k, gi])) if pacpos is not None and not np.isnan(pacpos[r, k]).any() else np.nan
                        at_resp = any(-10 <= e - k <= 40 for e in events_[r])          # the loss is followed by / sits at a life loss
                        evs.append({"rollout": r, "ghost": g, "step": k + 1, "pos0": float(P[r, k, gi, 0]), "pos1": float(P[r, k, gi, 1]),
                                    "at_life_loss": at_resp, "near_pacman": bool(d_pac < 6), "near_other_ghost": bool(others and min(others) < 5),
                                    "at_edge": bool(P[r, k, gi, 1] < 5 or P[r, k, gi, 1] > 59), "absent_steps": back if back > 0 else H - k,
                                    "permanent": back < 0, "returns_in_pen": bool(back > 0 and zone[r, k + back, gi])})
                        k += a.loss_gap
                    k += 1
        return evs

    gt_pac = None
    print(f"\n=== (e) in-maze disappearances: a detected ghost outside the pen zone, then undetected for {a.loss_gap}+ steps ===")
    print(f"{'world':14s}{'events':>8s}{'at life loss':>14s}{'near Pac-Man':>14s}{'near ghost':>12s}{'at edge':>9s}{'none of these':>15s}{'permanent':>11s}{'median absence':>16s}")
    for world, d, P, zone, pp, ev_ in (("ground truth", gt_det, gt_pos, gz, gt_pac, gt_ev), ("model", det, pos, mz, pac, m_ev)):
        evs = disappearances(d, P, zone, pp, ev_, valid if world == "ground truth" else np.ones((R, H), bool))
        n = max(len(evs), 1)
        f = lambda key: 100 * sum(e[key] for e in evs) / n
        none = 100 * sum(not (e["at_life_loss"] or e["near_pacman"] or e["near_other_ghost"] or e["at_edge"]) for e in evs) / n
        med = pct([e["absent_steps"] for e in evs if not e["permanent"]], 50)
        print(f"{world:14s}{len(evs):>8d}{f('at_life_loss'):>13.0f}%{f('near_pacman'):>13.0f}%{f('near_other_ghost'):>11.0f}%{f('at_edge'):>8.0f}%{none:>14.0f}%{f('permanent'):>10.0f}%{med:>16.0f}")
        rows.append({"part": "e_disappearances", "ghost": "all", "world": world, "key": "events", "n": len(evs), "value": none / 100, "median": med,
                     "p25": f("at_life_loss") / 100, "p75": f("near_pacman") / 100, "p90": f("permanent") / 100})
        if world == "model" and evs:
            unexpl = [e for e in evs if not e["at_life_loss"]]
            lost_steps = sum(e["absent_steps"] for e in unexpl)
            perm_steps = sum(e["absent_steps"] for e in unexpl if e["permanent"])
            print(f"  model, not at a life loss: {len(unexpl)} events = {lost_steps} absent ghost-steps ({lost_steps / (R * H):.2f} ghosts/step over the whole rollout); "
                  f"{100 * perm_steps / max(lost_steps, 1):.0f}% of those steps belong to ghosts that never come back")
            print("    by ghost: " + ", ".join(f"{g} {sum(e['ghost'] == g for e in unexpl)}" for g in G) +
                  f" | when: 1-130 {sum(e['step'] <= 130 for e in unexpl)}, 131-250 {sum(130 < e['step'] <= 250 for e in unexpl)}, 251-{H} {sum(e['step'] > 250 for e in unexpl)}")
            print(f"    near Pac-Man {sum(e['near_pacman'] for e in unexpl)}, near another ghost {sum(e['near_other_ghost'] for e in unexpl)}, at the tunnel edge {sum(e['at_edge'] for e in unexpl)}, "
                  f"returns inside the pen {sum(e['returns_in_pen'] for e in unexpl)}, permanent {sum(e['permanent'] for e in unexpl)}")
            write_csv(out / "residual_disappearances.csv", evs)
    write_csv(out / "residual_ghosts.csv", rows)
    print(f"\nwrote {out}/residual_ghosts.csv" + (" and residual_loss_events.csv" if events else ""))


if __name__ == "__main__":
    main()
