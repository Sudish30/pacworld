"""Ghost count over the rollout, as a ratio to each run's own ground truth, for DIAMOND next to our models.

Per rollout and bin: sum of detected ghosts in the model's frames / sum in the ground-truth frames (gated steps only,
as eval_rollouts). Rollouts are averaged per episode; the band is a bootstrap 95% CI over episodes. Also the detector
check: the same ratio over the first ghost_ratio_plot.early_steps steps, where the model has not yet drifted, and the
ground-truth detection level (detected ghosts per gated ground-truth frame).

  python eval/ghost_ratio_plot.py --config configs/stats.yaml --seed 0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from bootstrap_ci import load

ROOT = Path(__file__).resolve().parents[1]


def episode_ratio(df, lo, hi):
    """(episodes, value): mean over an episode's rollouts of sum(model ghosts) / sum(GT ghosts) in steps (lo, hi]."""
    sel = (df["step"] > lo) & (df["step"] <= hi) & df["gt"] & np.isfinite(df["ghost_count"]) & np.isfinite(df["ghost_count_gt"])
    out = {}
    for e in np.unique(df["episode"]):
        vals = []
        for s in np.unique(df["seed"][df["episode"] == e]):
            m = sel & (df["episode"] == e) & (df["seed"] == s)
            if df["ghost_count_gt"][m].sum() > 0:
                vals.append(df["ghost_count"][m].sum() / df["ghost_count_gt"][m].sum())
        if vals:
            out[int(e)] = float(np.mean(vals))
    return out


def ci(vals, rng, B):
    v = np.array(list(vals.values()))
    boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/stats.yaml")
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    g = cfg["ghost_ratio_plot"]
    rng = np.random.default_rng(a.seed)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.6))
    res = {}
    for name, label in g["runs"].items():
        df = load(ROOT / cfg["runs"][name])
        edges = list(range(0, g["horizon"] + 1, g["bin"]))
        pts = [ci(episode_ratio(df, lo, hi), rng, g["bootstrap"]) for lo, hi in zip(edges, edges[1:])]
        x = [hi for hi in edges[1:]]
        gated = df["gt"] & np.isfinite(df["ghost_count_gt"]) & (df["step"] <= g["horizon"])
        res[name] = {"early_ratio": ci(episode_ratio(df, 0, g["early_steps"]), rng, g["bootstrap"]),
                     "gt_ghosts_per_gated_frame": float(df["ghost_count_gt"][gated].mean()),
                     "bins": {int(k): v for k, v in zip(x, pts)}}
        ax.plot(x, [q[0] for q in pts], marker="o", ms=3, label=label)
        ax.fill_between(x, [q[1] for q in pts], [q[2] for q in pts], alpha=0.2)
        print(f"{name}: ratio over steps 1-{g['early_steps']} = {res[name]['early_ratio'][0]:.3f} "
              f"[{res[name]['early_ratio'][1]:.3f}, {res[name]['early_ratio'][2]:.3f}]; ground truth shows "
              f"{res[name]['gt_ghosts_per_gated_frame']:.2f} ghosts per gated frame; last bin {pts[-1][0]:.3f}")
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel(f"rollout step ({g['bin']}-step bins)")
    ax.set_ylabel("ghosts detected / own ground truth")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    (ROOT / g["out_png"]).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / g["out_png"], dpi=150)
    json.dump(res, open(ROOT / g["out_json"], "w"), indent=2)
    print(f"-> {g['out_png']}, {g['out_json']}")


if __name__ == "__main__":
    main()
