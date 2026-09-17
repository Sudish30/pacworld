"""Quantify 'freeze' (stall) behaviour in generated rollouts.

Two criteria, both calibrated against real frames from the same episodes:

  pixel   15+ consecutive steps whose frame is nearly identical to the previous
          one (mean absolute difference < --mad-tau over 0-255 pixels).
  motion  15+ consecutive steps in which the detected Pac-Man moves less than
          --move-tau px/step, i.e. the game state stops advancing even though
          pixels keep changing.

Calibration matters because the real game is never pixel-static (the power
pellets blink) and does contain genuine pauses: after a life is lost Ms. Pac-Man
stands still for ~20 steps. Running the same detector over the real frames gives
the false-positive rate for any (tau, min_len), and the minimum run length that
the real game never reaches.

  python eval/freeze_analysis.py --seed 0 --model model1
"""
import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from common import load_config  # noqa: E402
import detectors as D  # noqa: E402
from eval_rollouts import GATED, UNGATED, load_episodes, summarise, write_csv  # noqa: E402

_REF = None


def _init(cfg):
    global _REF
    _REF = D.load_reference(cfg)


def _pac_track(frames):
    """Pac-Man (row, col) per frame; None where undetected."""
    out = []
    for f in frames:
        d = _REF.sprites(f)
        p = d.get("pac")
        out.append(None if p is None else (p[0], p[1]))
    return out


def runs_below(values, tau, min_len):
    """Start indices (0-based) and lengths of runs where values < tau for >= min_len steps."""
    below = np.asarray(values) < tau
    out, i = [], 0
    while i < len(below):
        if below[i]:
            j = i
            while j < len(below) and below[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def step_motion(track, max_jump=20.0):
    """Per-step Pac-Man displacement; nan where either endpoint is missing or the move is a tunnel wrap."""
    m = np.full(len(track) - 1, np.nan)
    for k in range(1, len(track)):
        a, b = track[k - 1], track[k]
        if a is None or b is None:
            continue
        d = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        m[k - 1] = np.nan if d > max_jump else d
    return m


def calibrate(cfg, eps, pool):
    """Run both detectors over the real frames of the eval episodes."""
    real = [D.downsample_frames(e["native"]) if "native" in e else e["frames"] for e in eps]
    mad = [np.abs(np.diff(f.astype(np.int16), axis=0)).mean(axis=(1, 2, 3)) for f in real]
    tracks = pool.map(_pac_track, real)
    motion = [step_motion(t) for t in tracks]
    pauses = []
    for e, t in zip(eps, tracks):
        lives = e["ram"][:, 123].astype(int)
        pos = e["ram"][:, [10, 16]].astype(int)
        same = (np.diff(pos, axis=0) == 0).all(1)
        for i in np.nonzero(np.diff(lives) < 0)[0]:
            a, b = i, i
            while a > 0 and same[a - 1]:
                a -= 1
            while b < len(same) and same[b]:
                b += 1
            pauses.append(b - a + 1)
    return {"mad": np.concatenate(mad), "motion": np.concatenate([m[~np.isnan(m)] for m in motion]),
            "mad_per_ep": mad, "motion_per_ep": motion, "pauses": np.array(pauses), "tracks": tracks}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--model", default="model1")
    p.add_argument("--mad-tau", type=float, default=0.30, help="pixel criterion: MAD below this = 'nearly identical'")
    p.add_argument("--move-tau", type=float, default=0.35, help="motion criterion: px/step below this = not moving")
    p.add_argument("--min-len", type=int, default=15, help="run length that counts as a freeze (as specified)")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--recompute", action="store_true", help="ignore the cached Pac-Man tracks")
    a = p.parse_args()
    cfg = load_config(a.config)
    res = ROOT / cfg["out_dir"] / a.model
    preds = np.load(res / "preds_all.npy", mmap_mode="r")
    jobs = list(csv.DictReader(open(res / "jobs.csv")))
    eps = load_episodes(cfg)
    by_seed = {e["seed"]: e for e in eps}
    start = cfg["rollout"]["start_step"]

    cache = res / "freeze" / "tracks.npz"
    if cache.exists() and not a.recompute:
        z = np.load(cache, allow_pickle=True)
        cal, tracks, mad = z["cal"].item(), list(z["tracks"]), z["mad"]
        print(f"loaded cached tracks from {cache}")
    else:
        with Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
            print("calibrating on real frames ...")
            cal = calibrate(cfg, eps, pool)
            print("tracking Pac-Man in generated frames ...")
            tracks = pool.map(_pac_track, [np.asarray(preds[j]) for j in range(preds.shape[0])])
        mad = np.stack([np.abs(np.diff(np.asarray(preds[j], np.int16), axis=0)).mean(axis=(1, 2, 3)) for j in range(preds.shape[0])])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, cal=cal, tracks=np.array(tracks, dtype=object), mad=mad)
    motion = [step_motion(t) for t in tracks]

    # ---------------------------------------------------------------- calibration report
    real_pause = cal["pauses"]
    print("\n=== calibration on real frames (same 10 episodes) ===")
    print(f"real MAD between consecutive frames: min {cal['mad'].min():.3f}  p1 {np.percentile(cal['mad'],1):.3f}  "
          f"p5 {np.percentile(cal['mad'],5):.3f}  median {np.median(cal['mad']):.3f}")
    print(f"model MAD: min {mad.min():.3f}  p1 {np.percentile(mad,1):.3f}  p5 {np.percentile(mad,5):.3f}  median {np.median(mad):.3f}")
    print(f"real Pac-Man motion px/step: median {np.median(cal['motion']):.2f}  p10 {np.percentile(cal['motion'],10):.2f}  "
          f"fraction below move-tau={a.move_tau}: {100*(cal['motion']<a.move_tau).mean():.1f}%")
    print(f"real pause after a life loss (RAM position static): n={len(real_pause)}  min {real_pause.min()}  "
          f"median {int(np.median(real_pause))}  max {real_pause.max()} steps   (context window K={cfg['data']['context']})")
    print("\nfalse positives of each criterion on REAL frames, by required run length:")
    print(f"{'min_len':>8} {'pixel runs (episodes)':>24} {'motion runs (episodes)':>24}")
    for L in (15, 20, 30, 40, 50, 60, 80):
        pr = sum(len(runs_below(m, a.mad_tau, L)) for m in cal["mad_per_ep"])
        pe = sum(len(runs_below(m, a.mad_tau, L)) > 0 for m in cal["mad_per_ep"])
        mr = sum(len(runs_below(np.nan_to_num(m, nan=99), a.move_tau, L)) for m in cal["motion_per_ep"])
        me = sum(len(runs_below(np.nan_to_num(m, nan=99), a.move_tau, L)) > 0 for m in cal["motion_per_ep"])
        print(f"{L:>8} {pr:>15} ({pe}/10) {mr:>15} ({me}/10)")
    safe = next((L for L in range(15, 200) if all(len(runs_below(np.nan_to_num(m, nan=99), a.move_tau, L)) == 0
                                                  for m in cal["motion_per_ep"])), None)
    print(f"\nshortest run length the real game never reaches (motion criterion): {safe} steps")

    # ---------------------------------------------------------------- per-rollout detection
    rows = []
    for j, job in enumerate(jobs):
        ep = by_seed[int(job["episode"])]
        lives = ep["ram"][:, 123].astype(int)
        losses = np.nonzero(np.diff(lives) < 0)[0]
        pix = runs_below(mad[j], a.mad_tau, a.min_len)
        mot = runs_below(np.nan_to_num(motion[j], nan=99), a.move_tau, a.min_len)
        mot_safe = runs_below(np.nan_to_num(motion[j], nan=99), a.move_tau, safe)
        r = {"row": j, "episode": job["episode"], "seed": job["seed"], "episode_len": int(job["episode_len"]),
             "pixel_freezes": len(pix), "pixel_first_step": pix[0][0] + 2 if pix else "",
             "motion_freezes": len(mot), "motion_first_step": "", "motion_longest": 0,
             "frozen": bool(mot_safe), "stall_start_step": "", "stall_len": "", "stall_gt_step": "",
             "nearest_life_loss_step": "", "steps_from_life_loss": "", "gt_available": ""}
        if mot:
            r["motion_first_step"] = mot[0][0] + 2
            r["motion_longest"] = max(L for _, L in mot)
        if mot_safe:
            s0, L = max(mot_safe, key=lambda z: z[1])
            k = s0 + 2                      # 1-based rollout step where the stall begins
            i = start + k - 1               # index into the real episode
            r.update({"stall_start_step": k, "stall_len": L, "stall_gt_step": i,
                      "gt_available": i < int(job["episode_len"]) + 1})
            if len(losses):
                near = int(losses[np.argmin(np.abs(losses - i))])
                r["nearest_life_loss_step"] = near
                r["steps_from_life_loss"] = int(i - near)
        rows.append(r)
    out = res / "freeze"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "per_rollout.csv", rows)

    frozen = [r for r in rows if r["frozen"]]
    print(f"\n=== freezes in {len(rows)} {a.model} rollouts ===")
    print(f"pixel criterion (MAD < {a.mad_tau}, {a.min_len}+ steps): {sum(r['pixel_freezes'] > 0 for r in rows)}/{len(rows)} rollouts")
    print(f"motion criterion ({a.min_len}+ steps below {a.move_tau} px/step): {sum(r['motion_freezes'] > 0 for r in rows)}/{len(rows)} rollouts "
          f"-- but the real game itself trips this, see calibration")
    print(f"motion criterion, calibrated ({safe}+ steps): {len(frozen)}/{len(rows)} rollouts = {100*len(frozen)/len(rows):.0f}%")
    print(f"\n{'row':>3} {'episode':>10} {'sd':>2} {'stall@step':>10} {'len':>5} {'gt step':>8} {'life loss':>9} {'offset':>7} {'gt?':>5}")
    for r in rows:
        if not r["frozen"]:
            print(f"{r['row']:>3} {r['episode']:>10} {r['seed']:>2} {'-':>10}")
            continue
        print(f"{r['row']:>3} {r['episode']:>10} {r['seed']:>2} {r['stall_start_step']:>10} {r['stall_len']:>5} "
              f"{r['stall_gt_step']:>8} {r['nearest_life_loss_step']:>9} {r['steps_from_life_loss']:>+7} {str(r['gt_available']):>5}")
    off = [r["steps_from_life_loss"] for r in frozen if r["steps_from_life_loss"] != ""]
    if off:
        near = [o for o in off if abs(o) <= 30]
        print(f"\nstalls beginning within 30 steps of a ground-truth life loss: {len(near)}/{len(off)}  "
              f"(offsets: median {int(np.median(off))}, range {min(off)} to {max(off)})")

    # ---------------------------------------------------------------- split headline table
    per_step = list(csv.DictReader(open(res / "per_step.csv")))
    def conv(row):
        o = {}
        for k, v in row.items():
            if k == "model":
                o[k] = v
            elif v in ("True", "False"):
                o[k] = v == "True"
            elif k in ("episode", "seed", "step", "resp_event", "resp_hit", "ghost_count_ungated", "pellets_remaining"):
                o[k] = int(float(v)) if v not in ("", "nan") else 0
            else:
                o[k] = float(v) if v not in ("", "nan") else np.nan
        return o
    per_step = [conv(r) for r in per_step]
    keys_frozen = {(int(r["episode"]), int(r["seed"])) for r in frozen}
    groups = {"frozen": [r for r in per_step if (r["episode"], r["seed"]) in keys_frozen],
              "not frozen": [r for r in per_step if (r["episode"], r["seed"]) not in keys_frozen]}
    hs = cfg["rollout"]["gated_horizons"] + cfg["rollout"]["ungated_horizons"]
    summaries = {}
    for name, rs in groups.items():
        if not rs:
            continue
        summaries[name] = summarise(cfg, rs, f"{a.model} [{name}]")
        write_csv(out / f"summary_{name.replace(' ', '_')}.csv", summaries[name])
    print(f"\n=== headline metrics split by stall ({len(keys_frozen)} frozen / {len(rows) - len(keys_frozen)} not frozen rollouts) ===")
    head = f"{'metric':22s} " + "  ".join(f"{('h' + str(h)):>15s}" for h in hs)
    for name, summ in summaries.items():
        print(f"\n-- {name} ({len({(r['episode'], r['seed']) for r in groups[name]})} rollouts)")
        print(head)
        for metric in GATED + ["responsiveness_events"] + [m for m in UNGATED if m not in GATED]:
            cells = []
            for h in hs:
                s = [x for x in summ if x["metric"] == metric and x["horizon"] == h]
                cells.append(f"{s[0]['mean']:6.3f} ± {s[0]['std']:5.3f}" + ("*" if not s[0]["gated"] else "") if s and s[0]["n"] else "-")
            print(f"{metric:22s} " + "  ".join(f"{c:>15s}" for c in cells))
    print(f"\nwrote {out}/per_rollout.csv and split summaries  (* = ungated)")


if __name__ == "__main__":
    main()
