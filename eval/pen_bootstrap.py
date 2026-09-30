"""Pen-timer headline metrics with bootstrap 95% CIs over held-out episodes, originals alongside.

Per rollout: the longest run of consecutive steps with the pen occupied, by eval/pen_timer_analysis.py's own
class-agnostic occupancy test (pen-box pixels differing from the maze reference by > occ_resid in >= occ_pixels
pixels), on the run's saved frames (preds_all.npy, rows described by jobs.csv). The same test on the ground-truth
frames of each episode gives the real-game reference. Resampling unit = episode (with all its sampler seeds).
Reported: rollouts parked >= parked_len steps (count, fraction + CI), median longest pen stay (+ CI), and paired
differences between runs on the same episodes.

  python eval/pen_bootstrap.py --config configs/pen_stats.yaml --seed 0
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from common import load_config  # noqa: E402
from dataset import episode_path  # noqa: E402
import detectors as D  # noqa: E402
from pen_timer_analysis import geom  # noqa: E402


def longest(b):
    best = cur = 0
    for x in b:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def run_longest(ecfg, run_dir, H):
    """(episode seeds per rollout, longest model pen run per rollout, longest ground-truth run per episode), first H steps."""
    ref = D.load_reference(ecfg)
    G = geom(ref.size)
    box, bg = G["pen_box"], ref.bg64[G["pen_box"]]

    def occ(frames):
        f = np.asarray(frames)[:, box[0], box[1]].astype(float)
        return ((np.linalg.norm(f - bg[None], axis=-1) > G["occ_resid"]).sum((1, 2)) >= G["occ_pixels"])

    preds = np.load(run_dir / "preds_all.npy", mmap_mode="r")
    with open(run_dir / "jobs.csv") as fh:
        jobs = list(csv.DictReader(fh))
    start = ecfg["rollout"]["start_step"]
    H = min(H, preds.shape[1])
    eps = [int(j["episode"]) for j in jobs]
    ml = [longest(occ(preds[int(j["row"]), :H])) for j in jobs]
    gl = {}
    for sd in sorted(set(eps)):
        native = np.load(episode_path(ecfg["data"], sd, ROOT))["frames"][start:start + H]
        gl[sd] = longest(occ(D.downsample_frames(native, *D.frame_geometry(ecfg))))
    return np.array(eps), np.array(ml), gl


def boot_stats(eps, vals, fn, B, rng):
    ue = np.unique(eps)
    idx = {e: np.nonzero(eps == e)[0] for e in ue}
    bs = []
    for _ in range(B):
        pick = rng.choice(ue, len(ue), replace=True)
        bs.append(fn(vals[np.concatenate([idx[e] for e in pick])]))
    return float(fn(vals)), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/pen_stats.yaml")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--runs", nargs="*", help="subset of runs (default: all in the config)")
    a = p.parse_args()
    c = yaml.safe_load(open(ROOT / a.config))
    B, L = c["bootstrap"], c["parked_len"]
    names = a.runs or list(c["runs"])
    out = {}
    for H in c["horizons"]:
        res, per = {"runs": {}, "pairs": {}}, {}
        print(f"=== horizon {H}" + (" (the window of the originally reported numbers)" if H == 450 else " (supplementary)"))
        for name in names:
            rc = c["runs"][name]
            run_dir = ROOT / rc["dir"]
            if not (run_dir / "preds_all.npy").exists():
                print(f"{name}: no preds_all.npy, skipped")
                continue
            ecfg = load_config(ROOT / rc["config"])
            eps, ml, gl = run_longest(ecfg, run_dir, H)
            per[name] = (eps, ml)
            parked = (ml >= L).astype(float)
            pf, pci = boot_stats(eps, parked, np.mean, B, np.random.default_rng(a.seed))
            md, mci = boot_stats(eps, ml.astype(float), np.median, B, np.random.default_rng(a.seed))
            g = np.array(list(gl.values()))
            r = {"rollouts": int(len(ml)), "episodes": int(len(np.unique(eps))),
                 "parked_count": int(parked.sum()), "parked_frac": pf, "parked_frac_ci95": pci,
                 "median_longest": md, "median_longest_ci95": mci, "max_longest": int(ml.max()),
                 "gt_median_longest": float(np.median(g)), "gt_max_longest": int(g.max()), "gt_parked": int((g >= L).sum())}
            o = (c.get("original") or {}).get(name) if H == 450 else None
            if o:
                r["original"] = o
                r["reproduces_original"] = bool(r["parked_count"] == o["parked"] and abs(md - o["median"]) < 0.51)
            res["runs"][name] = r
            print(f"{name:22s} parked {r['parked_count']}/{r['rollouts']} = {pf:.3f} [{pci[0]:.3f}, {pci[1]:.3f}]  "
                  f"median longest {md:.1f} [{mci[0]:.1f}, {mci[1]:.1f}] (max {r['max_longest']})  "
                  f"gt median {r['gt_median_longest']:.0f} max {r['gt_max_longest']}"
                  + (f"  original {o['parked']}/30, {o['median']} -> {'REPRODUCED' if r['reproduces_original'] else 'DIFFERS'}" if o else ""), flush=True)
        for x, y in c["pairs"]:
            if x not in per or y not in per:
                continue
            (ex, mx), (ey, my) = per[x], per[y]
            ue = sorted(set(ex.tolist()) & set(ey.tolist()))
            d = np.array([np.mean(my[ey == e] >= L) - np.mean(mx[ex == e] >= L) for e in ue])
            rng = np.random.default_rng(a.seed)
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
            q = {"parked_frac_diff": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "episodes": len(ue)}
            res["pairs"][f"{y} - {x}"] = q
            print(f"  {y} - {x}: parked fraction {q['parked_frac_diff']:+.3f} [{q['ci95'][0]:+.3f}, {q['ci95'][1]:+.3f}]")
        out[str(H)] = res
    res = out
    od = ROOT / c["out"]
    od.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(od / "pen_bootstrap.json", "w"), indent=2)
    print(f"wrote {od}/pen_bootstrap.json")


if __name__ == "__main__":
    main()
