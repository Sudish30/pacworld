"""Bootstrap 95% CIs over held-out EPISODES for the headline rollout metrics, with the original numbers alongside.

Per rollout (episode x sampler seed) each metric is computed exactly as eval_rollouts.summarise does: the mean of the
last `window` gated steps up to the horizon, or hits/events for responsiveness. Rollouts are averaged per episode;
the bootstrap resamples episodes (with all their seeds). Paired differences between two runs use the same resampled
episodes for both. The "original" column is the mean over rollouts that summary.csv reports.

  python eval/bootstrap_ci.py --config configs/stats.yaml --seed 0
"""
import argparse
import json
from pathlib import Path

import csv

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


# ratios to the same run's own ground truth: comparable across frame geometries (DIAMOND's full-screen frames)
DERIVED = {"wall_iou_rel": ("wall_iou", "wall_iou_gt"), "ghost_count_rel": ("ghost_count", "ghost_count_gt")}


def load(path):
    """per_step.csv -> dict of numpy columns (bools and numbers parsed; empty cells -> NaN)."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    cols = {}
    for k in rows[0]:
        v = [r[k] for r in rows]
        if k in ("gt", "pac_detected"):
            cols[k] = np.array([x == "True" for x in v])
        elif k == "model":
            continue
        else:
            cols[k] = np.array([float(x) if x not in ("", "nan") else np.nan for x in v])
    return cols


def per_rollout(df, metric, h, w):
    """dict (episode, seed) -> the per-rollout value at horizon h (NaN if undefined), as eval_rollouts.summarise."""
    out = {}
    keys = np.stack([df["episode"], df["seed"]], 1)
    for key in np.unique(keys, axis=0):
        m = (df["episode"] == key[0]) & (df["seed"] == key[1]) & (df["step"] <= h)
        if metric == "responsiveness":
            ev = df["resp_event"][m & df["gt"]].sum()
            out[(int(key[0]), int(key[1]))] = df["resp_hit"][m].sum() / ev if ev > 0 else np.nan
        elif metric in DERIVED:
            num, den = DERIVED[metric]
            sel = m & (df["step"] > h - w) & df["gt"]
            v = df[num][sel] / df[den][sel]
            v = v[np.isfinite(v)]
            out[(int(key[0]), int(key[1]))] = v.mean() if len(v) else np.nan
        else:
            v = df[metric][m & (df["step"] > h - w) & df["gt"]]
            v = v[~np.isnan(v)]
            out[(int(key[0]), int(key[1]))] = v.mean() if len(v) else np.nan
    return out


def by_episode(r):
    eps = {}
    for (e, _), v in r.items():
        eps.setdefault(e, []).append(v)
    return {e: float(np.nanmean(v)) if not np.all(np.isnan(v)) else np.nan for e, v in eps.items()}


def boot(ep_values, eps, B, rng):
    """ep_values: dict episode -> per-episode mean. Returns point (mean over episodes) and 95% CI."""
    v = np.array([ep_values[e] for e in eps])
    ok = ~np.isnan(v)
    v, n = v[ok], ok.sum()
    idx = rng.integers(0, n, (B, n))
    bs = v[idx].mean(1)
    return float(v.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], int(n)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/stats.yaml")
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    c = yaml.safe_load(open(ROOT / a.config))
    B, w = c["bootstrap"], c["window"]
    dfs = {name: load(ROOT / f) for name, f in c["runs"].items() if (ROOT / f).exists()}
    for name, k in (c.get("pac_err_scale") or {}).items():
        if name in dfs:
            dfs[name]["pac_err"] = dfs[name]["pac_err"] * k        # 64px-equivalent units for every run
    missing = [n for n in c["runs"] if n not in dfs]
    out = {"missing_runs": missing, "runs": {}, "pairs": {}}
    per_ep = {}
    for name, df in dfs.items():
        out["runs"][name] = {}
        for m in c["metrics"]:
            for h in c["horizons"]:
                r = per_rollout(df, m, h, w)
                ep = by_episode(r)
                per_ep[(name, m, h)] = ep
                rng = np.random.default_rng(a.seed)
                pt, ci, n = boot(ep, sorted(ep), B, rng)
                vals = np.array(list(r.values()), float)
                out["runs"][name][f"{m}@{h}"] = {"mean_over_episodes": pt, "ci95": ci, "episodes": n,
                                                 "original_mean_over_rollouts": float(np.nanmean(vals)), "rollouts": int((~np.isnan(vals)).sum())}
    for x, y in c["pairs"]:
        if x not in dfs or y not in dfs:
            continue
        key = f"{y} - {x}"
        out["pairs"][key] = {}
        for m in c["metrics"]:
            for h in c["horizons"]:
                ex, ey = per_ep[(x, m, h)], per_ep[(y, m, h)]
                eps = sorted(set(ex) & set(ey))
                diff = {e: ey[e] - ex[e] for e in eps}
                rng = np.random.default_rng(a.seed)
                pt, ci, n = boot(diff, eps, B, rng)
                out["pairs"][key][f"{m}@{h}"] = {"diff": pt, "ci95": ci, "episodes": n, "excludes_0": ci[0] > 0 or ci[1] < 0}
    od = ROOT / c["out"]
    od.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(od / "bootstrap_ci.json", "w"), indent=2)
    lines = ["| run | " + " | ".join(f"{m}@{h}" for m in c["metrics"] for h in (450,)) + " |", "|---" * (1 + len(c["metrics"])) + "|"]
    for name, r in out["runs"].items():
        cells = [f"{r[f'{m}@450']['mean_over_episodes']:.3f} [{r[f'{m}@450']['ci95'][0]:.3f}, {r[f'{m}@450']['ci95'][1]:.3f}] (orig {r[f'{m}@450']['original_mean_over_rollouts']:.3f})" for m in c["metrics"]]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "| paired difference | " + " | ".join(f"{m}@450" for m in c["metrics"]) + " |", "|---" * (1 + len(c["metrics"])) + "|"]
    for key, r in out["pairs"].items():
        cells = [f"{r[f'{m}@450']['diff']:+.3f} [{r[f'{m}@450']['ci95'][0]:+.3f}, {r[f'{m}@450']['ci95'][1]:+.3f}]{' *' if r[f'{m}@450']['excludes_0'] else ''}" for m in c["metrics"]]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    open(od / "bootstrap_ci_450.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"missing runs: {missing}; wrote {od}/bootstrap_ci.json and bootstrap_ci_450.md (* = CI excludes 0)")


if __name__ == "__main__":
    main()
