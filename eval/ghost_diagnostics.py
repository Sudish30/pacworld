"""Inference-only diagnostics for the ghost fade in Model 1 rollouts.

Conditions (same held-out episodes x sampler seeds as the main eval):
  teacher-forced   the REAL context frames are fed at every step, so errors never
                   compound. Separates "cannot render ghosts" from "the
                   autoregressive loop destroys them".
  autoregressive   the model's own outputs are fed back, at 3 / 10 / 20 Euler steps.
                   If more sampler steps keep ghosts alive, the fade is sampler
                   error compounding rather than a learned failure.

Reported per condition and per step, over frames that RAM says should show four
normally-coloured ghosts:
  ghost pixel mass as a fraction of ground truth  (does ghost colour survive at all)
  per-ghost detection rate                        (which of the four dies first)

Also reports how often each ghost appears in the recorded data in its pure colour
versus the max-pooled blend, since pooling lifts red and orange towards the
background but leaves pink and cyan unchanged.

  python eval/ghost_diagnostics.py --seed 0
"""
import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from common import load_config  # noqa: E402
from dataset import to_float, to_uint8  # noqa: E402
import detectors as D  # noqa: E402
import metrics as M  # noqa: E402
from eval_rollouts import DiffusionPredictor, load_episodes, write_csv  # noqa: E402

_REF = None


def _init(cfg):
    global _REF
    _REF = D.load_reference(cfg)


def _analyse(frames):
    """Per frame: per-ghost detected (T, 4), total ghost pixel mass (T,), ghost positions (T, 4, 2), Pac-Man (T, 2)."""
    det = np.zeros((len(frames), len(D.GHOST_NAMES)), bool)
    mass = np.zeros(len(frames))
    pos = np.full((len(frames), len(D.GHOST_NAMES), 2), np.nan)
    pac = np.full((len(frames), 2), np.nan)
    for t, f in enumerate(frames):
        pres = _REF.pellet_presence(f)
        s = _REF.sprites(f, pres)
        for gi, g in enumerate(D.GHOST_NAMES):
            if s[g] is not None:
                det[t, gi] = True
                pos[t, gi] = s[g][:2]
        if s["pac"] is not None:
            pac[t] = s["pac"][:2]
        mass[t] = _REF.ghost_mass(f, pres)["total"]
    return det, mass, pos, pac


def run_condition(cfg, predictor, eps, device, seed, horizon, teacher, batch):
    K, start, size = cfg["data"]["context"], cfg["rollout"]["start_step"], cfg["data"]["size"]
    jobs = [(ei, s) for ei in range(len(eps)) for s in range(cfg["rollout"]["seeds"])]
    preds = np.zeros((len(jobs), horizon, size, size, 3), np.uint8)

    def ctx_at(indices, bat):
        return torch.stack([to_float(torch.from_numpy(eps[ei]["frames"][i - K:i]).permute(0, 3, 1, 2)).reshape(3 * K, size, size)
                            for (ei, _), i in zip(bat, indices)]).to(device)

    for b0 in range(0, len(jobs), batch):
        bat = jobs[b0:b0 + batch]
        ctx = ctx_at([start] * len(bat), bat)
        gens = [torch.Generator(device=device).manual_seed(seed * 100003 + eps[ei]["seed"] % 100003 + 7919 * s) for ei, s in bat]
        for k in range(horizon):
            acts = torch.tensor([[int(eps[ei]["actions"][(start + k - K + j) % len(eps[ei]["actions"])]) for j in range(K)]
                                 for ei, _ in bat], device=device)
            pred = predictor(ctx, acts, gens)
            preds[b0:b0 + len(bat), k] = to_uint8(pred).permute(0, 2, 3, 1).cpu().numpy()
            ctx = ctx_at([start + k + 1] * len(bat), bat) if teacher else torch.cat([ctx[:, 3:], pred], dim=1)
    return jobs, preds


def pooled_colour_stats(cfg, eps, stride=9):
    """Fraction of each ghost's pixels drawn in the pure colour vs the max-pooled blend.

    Reads the native-resolution recordings, since load_episodes() keeps only the 64x64 cache.
    """
    folder = ROOT / cfg["data"]["folder"]
    natives = [np.load(folder / f"ep_{e['seed']}.npz")["frames"] for e in eps]
    rows = []
    for g in D.GHOST_NAMES:
        pure = np.array(D.GHOSTS[g])
        pooled = np.array(D.pooled_variant(D.GHOSTS[g]))
        np_, npo = 0, 0
        for native in natives:
            for t in range(0, len(native), stride):
                x = native[t]
                np_ += int(((x == pure).all(-1)).sum())
                npo += int(((x == pooled).all(-1)).sum())
        same = tuple(pure) == tuple(pooled)
        rows.append({"ghost": g, "pure_rgb": tuple(int(v) for v in pure), "pooled_rgb": tuple(int(v) for v in pooled),
                     "pooled_changes_colour": not same, "px_pure": np_, "px_pooled": 0 if same else npo,
                     "frac_pooled": np.nan if same else npo / max(np_ + npo, 1)})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--horizon", type=int, default=450)
    p.add_argument("--batch", type=int, default=30)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--smooth", type=int, default=21)
    p.add_argument("--ctx-sigmas", type=float, nargs="*", default=[0.01, 0.03, 0.05, 0.1],
                   help="inference context noise levels to sweep at 3 sampler steps")
    a = p.parse_args()
    cfg = load_config(a.config)
    out = ROOT / cfg["out_dir"] / "ghost_diag"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, start, K = a.horizon, cfg["rollout"]["start_step"], cfg["data"]["context"]

    eps = load_episodes(cfg)
    shortest = min(len(e["actions"]) for e in eps)
    if start + H > shortest:
        raise SystemExit(f"horizon {H} needs {start + H} frames but the shortest episode has {shortest}")
    print(f"episodes: {[(e['seed'], len(e['actions'])) for e in eps]}")

    print("\n=== how the recorded data renders each ghost (max-pooling) ===")
    stats = pooled_colour_stats(cfg, eps)
    for r in stats:
        note = "unchanged by pooling" if not r["pooled_changes_colour"] else f"{100 * r['frac_pooled']:.1f}% of pixels are the pooled blend"
        print(f"  {r['ghost']:7s} pure {str(r['pure_rgb']):>16s} -> pooled {str(r['pooled_rgb']):>16s}   {note}")
    write_csv(out / "pooled_colours.csv", stats)

    predictor = DiffusionPredictor(cfg, device)
    print(f"\npredictor: {predictor.name}")
    from model1 import karras_schedule
    dd = predictor.d

    conditions = [("teacher-forced, 3 steps", True, 3, 0.0), ("autoregressive, 3 steps", False, 3, 0.0),
                  ("autoregressive, 10 steps", False, 10, 0.0), ("autoregressive, 20 steps", False, 20, 0.0)]
    conditions += [(f"autoregressive, 3 steps, ctx sigma {sg}", False, 3, sg) for sg in a.ctx_sigmas]

    with Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
        # ground truth, per rollout (each rollout inherits its episode's ground truth)
        gt = pool.map(_analyse, [e["frames"][start:start + H] for e in eps])
        elig = [M.ghost_eligibility(e["ram"], cfg)[start:start + H] for e in eps]
        results = {}
        for name, teacher, steps, ctx_sigma in conditions:
            predictor.sigmas = karras_schedule(steps, dd["sigma_min"], dd["sigma_max"], dd["rho"], device)
            predictor.ctx_sigma = float(ctx_sigma)
            import time
            t0 = time.time()
            jobs, preds = run_condition(cfg, predictor, eps, device, a.seed, H, teacher, a.batch)
            print(f"\n{name}: {len(jobs)} rollouts x {H} steps sampled in {time.time() - t0:.0f}s; analysing frames ...")
            per = pool.map(_analyse, [preds[j] for j in range(preds.shape[0])])
            results[name] = {"jobs": jobs, "det": np.stack([x[0] for x in per]), "mass": np.stack([x[1] for x in per]),
                             "pos": np.stack([x[2] for x in per]), "pac": np.stack([x[3] for x in per])}
            del preds

    jobs = results[conditions[0][0]]["jobs"]
    ep_of = np.array([ei for ei, _ in jobs])
    E = np.stack([elig[ei] for ei in ep_of])                       # (30, H) eligible steps
    gt_det = np.stack([gt[ei][0] for ei in ep_of])                 # (30, H, 4)
    gt_mass = np.stack([gt[ei][1] for ei in ep_of])                # (30, H)
    raw = {"episode_seed": np.array([eps[ei]["seed"] for ei in ep_of]), "rollout_seed": np.array([s_ for _, s_ in jobs]),
           "eligible": E, "gt_det": gt_det, "gt_mass": gt_mass, "gt_pos": np.stack([gt[ei][2] for ei in ep_of]),
           "conditions": np.array(list(results)), "start": start, "horizon": H}
    for ci, name in enumerate(results):
        for key in ("det", "mass", "pos", "pac"):
            raw[f"c{ci}_{key}"] = results[name][key]
    np.savez_compressed(out / "raw.npz", **raw)

    def ratio_curve(mass):
        num = np.where(E, mass, 0).sum(0)
        den = np.where(E, gt_mass, 0).sum(0)
        return np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)

    def survival(det, gi):
        n = E.sum(0)
        return np.where(n > 0, np.where(E, det[:, :, gi], 0).sum(0) / np.maximum(n, 1), np.nan)

    def smooth(v, w):
        k = np.ones(w) / w
        pad = np.pad(v, (w // 2, w // 2), mode="edge")
        return np.convolve(np.nan_to_num(pad, nan=np.nanmean(v)), k, mode="valid")[:len(v)]

    rows = []
    for name in results:
        rc = ratio_curve(results[name]["mass"])
        for k in range(H):
            row = {"condition": name, "step": k + 1, "mass_ratio": rc[k],
                   "model_mass": float(np.where(E[:, k], results[name]["mass"][:, k], np.nan)[E[:, k]].mean()) if E[:, k].any() else np.nan,
                   "gt_mass": float(gt_mass[E[:, k], k].mean()) if E[:, k].any() else np.nan,
                   "eligible_rollouts": int(E[:, k].sum())}
            for gi, g in enumerate(D.GHOST_NAMES):
                row[f"surv_{g}"] = survival(results[name]["det"], gi)[k]
                row[f"gt_surv_{g}"] = survival(gt_det, gi)[k]
            rows.append(row)
    write_csv(out / "per_step.csv", rows)

    marks = [15, 50, 150, 300, 450]
    print(f"\n=== ghost pixel mass as a fraction of ground truth (mean over {a.smooth} steps) ===")
    print(f"{'condition':26s}" + "".join(f"{('step ' + str(m)):>12s}" for m in marks))
    for name in results:
        rc = smooth(ratio_curve(results[name]["mass"]), a.smooth)
        print(f"{name:26s}" + "".join(f"{100 * rc[m - 1]:11.0f}%" for m in marks))
    print(f"\n=== per-ghost detection rate across {len(jobs)} rollouts (eligible steps only) ===")
    for name in list(results) + ["ground truth"]:
        det = gt_det if name == "ground truth" else results[name]["det"]
        print(f"  {name}")
        for gi, g in enumerate(D.GHOST_NAMES):
            sv = smooth(survival(det, gi), a.smooth)
            print(f"    {g:7s}" + "".join(f"{100 * sv[m - 1]:11.0f}%" for m in marks))

    # ---------------------------------------------------------------- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = np.arange(1, H + 1)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i, name in enumerate(results):
        sweep = "ctx sigma" in name
        ax.plot(steps, 100 * smooth(ratio_curve(results[name]["mass"]), a.smooth), label=name, color=f"C{i}",
                lw=1.4 if sweep else 1.9, ls="--" if sweep else "-")
    ax.axhline(100, color="k", ls=":", lw=1, label="ground truth")
    ax.set_xlabel("rollout step (15 steps = 1 s)")
    ax.set_ylabel("ghost pixel mass, % of ground truth")
    ax.set_title("Ghost colour surviving the rollout, Model 1 EMA @100k")
    ax.set_ylim(0, 130)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "ghost_mass_fraction.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for gi, (g, ax) in enumerate(zip(D.GHOST_NAMES, axes)):
        for i, name in enumerate(results):
            if "ctx sigma" in name:
                continue
            ax.plot(steps, 100 * smooth(survival(results[name]["det"], gi), a.smooth), color=f"C{i}", lw=1.6,
                    label=name if gi == 0 else None)
        ax.plot(steps, 100 * smooth(survival(gt_det, gi), a.smooth), color="k", ls="--", lw=1.2,
                label="ground truth" if gi == 0 else None)
        pooled = D.pooled_variant(D.GHOSTS[g]) != D.GHOSTS[g]
        ax.set_title(f"{g}{'  (blended by pooling)' if pooled else '  (colour unchanged)'}", fontsize=10)
        ax.set_xlabel("rollout step")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("rollouts where this ghost is detected, %")
    fig.legend(loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "per_ghost_survival.png", dpi=140)
    plt.close(fig)
    print(f"\nwrote {out}/ghost_mass_fraction.png, per_ghost_survival.png, per_step.csv")


if __name__ == "__main__":
    main()
