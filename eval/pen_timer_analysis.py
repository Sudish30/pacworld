"""Test the pen-timer hypothesis for the ghost fade.

Hypothesis: ghosts are lost because the pen release (and the post-death reset)
runs on timers that cannot be read from a 4-frame context, so ghosts that go
through the pen are the ones that disappear. Alternative: generic compounding
error, in which case ghosts roaming the maze fade just as fast.

Uses ground-truth RAM (ghost x at bytes 6-9, y at 12-15; pen at x=88, y=80 with
the door leading up to y=50) to tag every ghost at every step as in-pen or
in-maze, and reads the per-rollout detections saved by ghost_diagnostics.py.

  python eval/pen_timer_analysis.py --seed 0
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

RAM_X = {"orange": 6, "cyan": 7, "pink": 8, "red": 9}
RAM_Y = {"orange": 12, "cyan": 13, "pink": 14, "red": 15}
PEN_X, PEN_Y, DOOR_Y = 88, 80, 50
PEN_BOX = (slice(28, 37), slice(26, 39))     # pen interior at 64x64, around RAM (88, 80) -> pixel (32, 32)
SPAWN_PX = (38.5, 31.9)                      # Pac-Man respawn point at 64x64, from RAM (88, 98)
OCC_RESID, OCC_PIXELS = 40, 6                # pen counts as occupied when >= 6 pixels differ from background by > 40


def in_pen(ram, g):
    """True while the ghost is inside the pen or travelling up the door shaft."""
    x, y = ram[:, RAM_X[g]].astype(int), ram[:, RAM_Y[g]].astype(int)
    return (np.abs(x - PEN_X) <= 6) & (y > DOOR_Y + 5) & (y <= PEN_Y + 4)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--conditions", nargs="*", default=["autoregressive, 3 steps", "autoregressive, 10 steps"])
    p.add_argument("--death-before", type=int, default=130)
    p.add_argument("--smooth", type=int, default=21)
    a = p.parse_args()
    cfg = load_config(a.config)
    out = ROOT / cfg["out_dir"] / "ghost_diag"
    z = np.load(out / "raw.npz", allow_pickle=True)
    names = list(z["conditions"])
    start, H = int(z["start"]), int(z["horizon"])
    R = len(z["episode_seed"])
    rams = {int(s): np.load(episode_path(cfg["data"], int(s), ROOT))["ram"] for s in np.unique(z["episode_seed"])}

    # per rollout ground-truth state
    G = D.GHOST_NAMES
    pen = np.zeros((R, H, 4), bool)          # ghost is in the pen at this step
    been = np.zeros((R, H, 4), bool)         # ghost has been in the pen at some point since the rollout began
    valid = np.zeros((R, H), bool)           # ghosts are drawn in their normal colours (not frightened, not death animation)
    first_loss = np.full(R, -1)
    for r in range(R):
        ram = rams[int(z["episode_seed"][r])]
        seg = ram[start:start + H]
        lives = ram[:, cfg["gating"]["lives_ram"]].astype(int)
        losses = np.nonzero(np.diff(lives) < 0)[0] + 1                 # frame index of the first frame of the new life
        v = seg[:, cfg["gating"]["frightened_ram"]] == 0
        for t in losses:
            k = t - start
            v[max(0, k - cfg["gating"]["death_anim_steps"]):max(0, k + 1)] = False
        valid[r] = v
        inroll = [t - start for t in losses if 0 <= t - start < H]
        first_loss[r] = inroll[0] if inroll else -1
        for gi, g in enumerate(G):
            ip = in_pen(ram, g)[start:start + H]
            pen[r, :, gi] = ip
            been[r, :, gi] = np.maximum.accumulate(ip)
    gt_det = z["gt_det"]

    print(f"rollouts: {R}; ground-truth life loss inside the 450-step window: {(first_loss >= 0).sum()}/{R}; "
          f"before step {a.death_before}: {((first_loss >= 0) & (first_loss < a.death_before)).sum()}/{R}")
    print(f"ghost-steps in pen: {100 * pen[valid].mean():.1f}% of valid ghost-steps; at rollout step 1: "
          f"{[f'{g} {100 * pen[:, 0, gi].mean():.0f}%' for gi, g in enumerate(G)]}")

    bins = [(1, 60), (61, 130), (131, 250), (251, 450)]
    rows = []

    def rate(det, mask):
        n = mask.sum()
        return (det[mask].mean() if n else np.nan), int(n)

    for cname in a.conditions:
        ci = names.index(cname)
        det = z[f"c{ci}_det"]
        print(f"\n=== {cname}: detection rate by ground-truth state (model / ground-truth ceiling) ===")
        print(f"{'state':34s}" + "".join(f"{f'steps {lo}-{hi}':>22s}" for lo, hi in bins))
        states = {
            "in pen now": lambda: pen,
            "in maze now": lambda: ~pen,
            "in maze, never penned since start": lambda: ~pen & ~been,
            "in maze, released during rollout": lambda: ~pen & been,
        }
        for sname, fn in states.items():
            cells = []
            for lo, hi in bins:
                m = fn()[:, lo - 1:hi] & valid[:, lo - 1:hi, None]
                rm, n = rate(det[:, lo - 1:hi], m)
                rg, _ = rate(gt_det[:, lo - 1:hi], m)
                cells.append(f"{100 * rm:4.0f}% /{100 * rg:4.0f}% n={n:<5d}" if n else f"{'-':>22s}")
                rows.append({"condition": cname, "state": sname, "steps": f"{lo}-{hi}", "model_detect": rm, "gt_detect": rg, "n": n})
            print(f"{sname:34s}" + "".join(f"{c:>22s}" for c in cells))
        print("  per ghost, in maze & never penned since start (the cleanest 'no timer involved' group):")
        for gi, g in enumerate(G):
            cells = []
            for lo, hi in bins:
                m = (~pen & ~been)[:, lo - 1:hi, gi] & valid[:, lo - 1:hi]
                n = m.sum()
                cells.append(f"{100 * det[:, lo - 1:hi, gi][m].mean():4.0f}% n={n:<5d}" if n else f"{'-':>14s}")
            print(f"    {g:7s}" + "".join(f"{c:>22s}" for c in cells))

    # ------------------------------------------------ split by early ground-truth death; event-aligned curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def smooth(v, w):
        k = np.ones(w) / w
        pad = np.pad(np.nan_to_num(v, nan=np.nanmean(v)), (w // 2, w // 2), mode="edge")
        return np.convolve(pad, k, mode="valid")[:len(v)]

    early = (first_loss >= 0) & (first_loss < a.death_before)
    groups = {f"life lost before step {a.death_before} (n={early.sum()})": early,
              f"no life lost before step {a.death_before} (n={(~early).sum()})": ~early}
    gt_mass = z["gt_mass"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 4.8))
    steps = np.arange(1, H + 1)
    print(f"\n=== ghost pixel mass, % of ground truth, split by early ground-truth life loss ===")
    marks = [50, 100, 130, 200, 300, 450]
    print(f"{'condition / group':62s}" + "".join(f"{('step ' + str(m)):>10s}" for m in marks))
    for ci_, cname in enumerate(a.conditions):
        ci = names.index(cname)
        mass = z[f"c{ci}_mass"]
        for gi_, (gname, sel) in enumerate(groups.items()):
            num = np.where(valid[sel], mass[sel], 0).sum(0)
            den = np.where(valid[sel], gt_mass[sel], 0).sum(0)
            curve = smooth(np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan), a.smooth)
            axes[0].plot(steps, 100 * curve, color=f"C{gi_}", ls="-" if ci_ == 0 else "--", lw=1.8,
                         label=f"{cname}: {gname}")
            print(f"{(cname + ': ' + gname):62s}" + "".join(f"{100 * curve[m - 1]:9.0f}%" for m in marks))
    axes[0].axvline(a.death_before, color="gray", lw=0.8, ls=":")
    axes[0].axhline(100, color="k", lw=0.8, ls=":")
    axes[0].set_title("Ghost mass by whether ground truth loses a life early")
    axes[0].set_xlabel("rollout step")
    axes[0].set_ylabel("ghost pixel mass, % of ground truth")
    axes[0].set_ylim(0, 130)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7.5)

    # event-aligned: x = steps relative to the first ground-truth life loss in the rollout
    W0, W1 = 120, 250
    print(f"\n=== ghost mass aligned on the first ground-truth life loss (rollouts with one in the window) ===")
    for ci_, cname in enumerate(a.conditions):
        ci = names.index(cname)
        mass = z[f"c{ci}_mass"]
        num, den = np.zeros(W0 + W1), np.zeros(W0 + W1)
        used = 0
        for r in range(R):
            if first_loss[r] < 0:
                continue
            used += 1
            for j, k in enumerate(range(first_loss[r] - W0, first_loss[r] + W1)):
                if 0 <= k < H and valid[r, k]:
                    num[j] += mass[r, k]
                    den[j] += gt_mass[r, k]
        curve = smooth(np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan), a.smooth)
        x = np.arange(-W0, W1)
        axes[1].plot(x, 100 * curve, lw=1.8, ls="-" if ci_ == 0 else "--", color="C3", label=f"{cname} (n={used})")
        print(f"{cname:30s}" + "".join(f"  t{off:+4d}: {100 * curve[off + W0]:4.0f}%" for off in (-100, -50, -10, 20, 60, 120, 200)))
    axes[1].axvline(0, color="k", lw=1)
    axes[1].axhline(100, color="k", lw=0.8, ls=":")
    axes[1].set_title("Ghost mass aligned on the first ground-truth life loss")
    axes[1].set_xlabel("steps relative to ground-truth life loss")
    axes[1].set_ylim(0, 130)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    # ------------------------------------------------ are the lost ghosts parked in the pen? (class-agnostic occupancy)
    # Ghosts stacked in the pen blend into a mixed colour that no ghost class matches, so neither the detector nor
    # the ghost-mass measure can see them. Occupancy looks at the pen interior directly: any non-background pixels.
    preds_path = ROOT / cfg["out_dir"] / "model1" / "preds_all.npy"
    if preds_path.exists() and "autoregressive, 3 steps" in names:
        ref = D.load_reference(cfg)
        preds = np.load(preds_path, mmap_mode="r")
        ci = names.index("autoregressive, 3 steps")
        bgbox = ref.bg64[PEN_BOX]

        def occ(frame):
            return (np.linalg.norm(frame[PEN_BOX].astype(float) - bgbox, axis=-1) > OCC_RESID).sum() >= OCC_PIXELS

        check = np.array([ref.ghost_mass(np.asarray(preds[0, k]))["total"] for k in range(0, H, 50)])
        assert np.allclose(check, z[f"c{ci}_mass"][0, 0:H:50]), "preds_all.npy is not the same rollouts as the AR-3 condition"
        m_occ = np.zeros((R, H), bool)
        g_occ = np.zeros((R, H), bool)
        gt_frames = {}
        for r in range(R):
            sd = int(z["episode_seed"][r])
            if sd not in gt_frames:
                gt_frames[sd] = D.downsample_frames(np.load(episode_path(cfg["data"], sd, ROOT))["frames"][start:start + H])
            g_occ[r] = [occ(f) for f in gt_frames[sd]]
            m_occ[r] = [occ(np.asarray(preds[r, k])) for k in range(H)]
        count = z[f"c{ci}_det"].sum(-1)

        def longest(b):
            best = cur = 0
            for x in b:
                cur = cur + 1 if x else 0
                best = max(best, cur)
            return best

        def runs(b, min_len):
            out_, i = [], 0
            while i < len(b):
                if b[i]:
                    j = i
                    while j < len(b) and b[j]:
                        j += 1
                    if j - i >= min_len:
                        out_.append((i, j - i))
                    i = j
                else:
                    i += 1
            return out_

        print(f"\n=== pen occupancy, class-agnostic (autoregressive, 3 steps) ===")
        print(f"{'steps':>10s} {'ground truth':>14s} {'model':>8s}")
        for lo, hi in bins:
            m = valid[:, lo - 1:hi]
            print(f"{lo:>4d}-{hi:<5d} {100 * g_occ[:, lo - 1:hi][m].mean():>13.0f}% {100 * m_occ[:, lo - 1:hi][m].mean():>7.0f}%")
        late, few = valid[:, 130:], count[:, 130:] <= 2
        print(f"model frames after step 130 showing <=2 ghosts: {100 * few[late].mean():.0f}%; pen occupied in "
              f"{100 * m_occ[:, 130:][late & few].mean():.0f}% of them (ground truth, same frames: {100 * g_occ[:, 130:][late & few].mean():.0f}%)")
        print(f"model frames after step 130 showing >=3 ghosts: pen occupied in {100 * m_occ[:, 130:][late & ~few].mean():.0f}%")
        gl, ml = [longest(g_occ[r]) for r in range(R)], [longest(m_occ[r]) for r in range(R)]
        print(f"longest continuous pen occupancy per rollout: ground truth median {int(np.median(gl))}, max {max(gl)} steps | "
              f"model median {int(np.median(ml))}, max {max(ml)} steps")
        print(f"rollouts with the pen occupied 200+ consecutive steps: model {sum(x >= 200 for x in ml)}/{R}, ground truth {sum(x >= 200 for x in gl)}/{R}")

        # model-world respawns: Pac-Man jumps to the spawn point
        pac = z[f"c{ci}_pac"]
        respawns = []
        for r in range(R):
            ev = []
            for k in range(1, H):
                a_, b_ = pac[r, k - 1], pac[r, k]
                if np.isnan(a_).any() or np.isnan(b_).any():
                    continue
                if np.hypot(*(b_ - a_)) > 6 and np.hypot(b_[0] - SPAWN_PX[0], b_[1] - SPAWN_PX[1]) < 2.5:
                    ev.append(k)
            respawns.append(ev)
        gt_losses = []
        for r in range(R):
            ram = rams[int(z["episode_seed"][r])]
            lo_ = np.nonzero(np.diff(ram[:, cfg["gating"]["lives_ram"]].astype(int)) < 0)[0] + 1 - start
            gt_losses.append([int(k) for k in lo_ if 0 <= k < H])
        print(f"\nmodel-world respawns (Pac-Man jumps to the spawn point): {sum(len(e) for e in respawns)} in {R} rollouts, "
              f"{sum(len(e) > 0 for e in respawns)}/{R} rollouts have at least one | ground-truth life losses in window: {sum(len(e) for e in gt_losses)}")
        long_runs = [(r, s0, L) for r in range(R) for s0, L in runs(m_occ[r], 100)]
        linked = sum(any(-40 <= s0 - k <= 15 for k in respawns[r]) for r, s0, L in long_runs)
        linked_gt = sum(any(-40 <= s0 - k <= 15 for k in gt_losses[r]) for r, s0, L in long_runs)
        print(f"pen-occupancy runs of 100+ steps in the model: {len(long_runs)}; beginning within 40 steps after a model-world respawn: "
              f"{linked}/{len(long_runs)}; after a GROUND-TRUTH life loss: {linked_gt}/{len(long_runs)}")
        first_resp = [e[0] for e in respawns if e]
        if first_resp:
            print(f"first model-world respawn: median step {int(np.median(first_resp))} (range {min(first_resp)}-{max(first_resp)}); "
                  f"first ground-truth life loss: median step {int(np.median([e[0] for e in gt_losses if e]))}")

        # release hazard after a respawn: for the pen-occupancy run that begins within [-40, +15] steps of each
        # respawn, the lag (steps since the run began) at which the pen empties, censored at the horizon.
        # hazard in a lag bin = releases in the bin / runs still occupied when the bin starts.
        def dwell_after(events, occ):
            out_ = []
            for r in range(R):
                for k in events[r]:
                    cands = [(s0, L) for s0, L in runs(occ[r], 1) if -40 <= s0 - k <= 15]
                    if cands:
                        s0, L = cands[0]
                        out_.append((L, s0 + L >= H))          # (length, censored by the horizon)
            return out_

        hz_bins = [(1, 30), (31, 60), (61, 90), (91, 150), (151, 450)]
        hazard_rows = []
        print(f"\n=== pen release hazard after a respawn (releases / runs still occupied at the start of the bin) ===")
        print(f"{'':22s}" + "".join(f"{f'lag {lo}-{hi}':>16s}" for lo, hi in hz_bins) + f"{'n runs':>8s}{'median dwell':>14s}")
        for label, ev, oc in (("model (own respawns)", respawns, m_occ), ("ground truth (real)", gt_losses, g_occ)):
            d = dwell_after(ev, oc)
            cells = []
            for lo, hi in hz_bins:
                at_risk = sum(1 for L, c in d if L >= lo)
                released = sum(1 for L, c in d if lo <= L <= hi and not c)
                h = released / at_risk if at_risk else np.nan
                cells.append(f"{100 * h:5.0f}% ({released}/{at_risk})" if at_risk else f"{'-':>16s}")
                hazard_rows.append({"metric": "release_hazard", "group": label, "bin": f"{lo}-{hi}", "value": h,
                                    "released": released, "at_risk": at_risk})
            med = int(np.median([L for L, _ in d])) if d else -1
            print(f"{label:22s}" + "".join(f"{c:>16s}" for c in cells) + f"{len(d):>8d}{med:>14d}")
            hazard_rows.append({"metric": "dwell_after_respawn_median", "group": label, "bin": "", "value": med, "released": len(d), "at_risk": len(d)})
            hazard_rows.append({"metric": "dwell_after_respawn_censored", "group": label, "bin": "", "value": sum(c for _, c in d), "released": len(d), "at_risk": len(d)})

        # machine-readable decision metrics for eval/compare_runs.py
        metrics = [{"metric": "pen_occupancy", "group": "ground truth", "bin": f"{lo}-{hi}", "value": g_occ[:, lo - 1:hi][valid[:, lo - 1:hi]].mean()} for lo, hi in bins]
        metrics += [{"metric": "pen_occupancy", "group": "model", "bin": f"{lo}-{hi}", "value": m_occ[:, lo - 1:hi][valid[:, lo - 1:hi]].mean()} for lo, hi in bins]
        metrics += [{"metric": "longest_pen_run_median", "group": "ground truth", "bin": "", "value": float(np.median(gl))},
                    {"metric": "longest_pen_run_median", "group": "model", "bin": "", "value": float(np.median(ml))},
                    {"metric": "longest_pen_run_max", "group": "ground truth", "bin": "", "value": max(gl)},
                    {"metric": "longest_pen_run_max", "group": "model", "bin": "", "value": max(ml)},
                    {"metric": "rollouts_pen_200plus", "group": "ground truth", "bin": "", "value": sum(x >= 200 for x in gl)},
                    {"metric": "rollouts_pen_200plus", "group": "model", "bin": "", "value": sum(x >= 200 for x in ml)},
                    {"metric": "respawns", "group": "model", "bin": "", "value": sum(len(e) for e in respawns)},
                    {"metric": "respawns", "group": "ground truth", "bin": "", "value": sum(len(e) for e in gt_losses)},
                    {"metric": "pen_runs_100plus", "group": "model", "bin": "", "value": len(long_runs)},
                    {"metric": "pen_runs_100plus_after_own_respawn", "group": "model", "bin": "", "value": linked}]
        for m in metrics:                                     # one column set for every row
            m.setdefault("released", ""), m.setdefault("at_risk", "")
        write_csv(out / "pen_metrics.csv", metrics + hazard_rows)

        axes[2].plot(steps, 100 * smooth(np.nanmean(np.where(valid, m_occ, np.nan), axis=0), a.smooth),
                     color="C1", lw=1.9, label="model (autoregressive, 3 steps)")
        axes[2].plot(steps, 100 * smooth(np.nanmean(np.where(valid, g_occ, np.nan), axis=0), a.smooth), color="k", ls="--", lw=1.4, label="ground truth")
        axes[2].set_title("Pen occupied (any non-background pixels in the pen)")
        axes[2].set_xlabel("rollout step")
        axes[2].set_ylabel("rollouts with the pen occupied, %")
        axes[2].set_ylim(0, 105)
        axes[2].grid(alpha=0.3)
        axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "pen_timer.png", dpi=140)
    write_csv(out / "pen_state_detection.csv", rows)
    print(f"\nwrote {out}/pen_timer.png and pen_state_detection.csv")


if __name__ == "__main__":
    main()
