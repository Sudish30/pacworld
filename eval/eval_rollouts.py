"""Autoregressive rollout evaluation of a world model on held-out episodes.

For the N longest held-out episodes and S sampler seeds, the context is taken at
`start_step`, the recorded actions are replayed through the model for `horizon`
steps, and every frame is scored against the recorded ground truth (frames + RAM,
bit-identical to an emulator replay by the alignment check). Metrics are reported
at the gated horizons (need ground truth) and, for ground-truth-free metrics only,
at the ungated horizons.

  python eval/eval_rollouts.py --seed 0 --model model1
  python eval/eval_rollouts.py --seed 0 --model model0
  python eval/eval_rollouts.py --seed 0 --plot eval/results/model1 eval/results/model0
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from common import load_config  # noqa: E402
from dataset import History, context_offsets, episode_files, to_uint8  # noqa: E402
import detectors as D  # noqa: E402
import metrics as M  # noqa: E402

GATED = ["pac_err", "pac_detected", "ghost_count", "ghost_count_gt", "ghost_abs_err", "pellet_iou", "wall_iou",
         "wall_iou_gt", "pellets_remaining", "pellets_remaining_gt", "responsiveness"]
UNGATED = ["ghost_count_ungated", "wall_iou", "pac_detected", "pellets_remaining"]


# ----------------------------------------------------------------------------- predictors
class DiffusionPredictor:
    def __init__(self, cfg, device):
        from model1 import build_model, karras_schedule
        c = cfg["model1"]
        ck = torch.load(ROOT / c["checkpoint"], map_location=device)
        self.model = build_model(ck["cfg"]).to(device).eval()
        self.model.load_state_dict(ck["ema"])
        self.step = ck["step"]
        self.offsets = context_offsets(ck["cfg"]["data"])     # the context layout the checkpoint was trained with
        self.d = ck["cfg"]["diffusion"]
        self.sigmas = karras_schedule(c["sampler_steps"], self.d["sigma_min"], self.d["sigma_max"], self.d["rho"], device)
        self.ctx_sigma = float(c["ctx_sigma"])
        self.bf16 = c["bf16"] and device.type == "cuda"
        self.device = device
        self.name = f"model1 (EMA step {self.step}, {c['sampler_steps']}-step Euler)"

    @torch.inference_mode()
    def __call__(self, ctx, acts, gens):
        B = ctx.shape[0]
        noise = torch.stack([torch.randn(3, *ctx.shape[2:], device=self.device, generator=g) for g in gens])
        x = noise * self.sigmas[0]
        ctx_sigma = torch.full((B,), self.ctx_sigma, device=self.device)
        ctx_in = ctx if self.ctx_sigma == 0 else ctx + self.ctx_sigma * torch.stack(
            [torch.randn(ctx.shape[1:], device=self.device, generator=g) for g in gens])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            for i in range(len(self.sigmas) - 1):
                s, s_next = self.sigmas[i], self.sigmas[i + 1]
                d = (x - self.model(x, s.expand(B), ctx_in, acts, ctx_sigma)) / s
                x = x + (s_next - s) * d
        return x.float().clamp(-1, 1)


class Model0Predictor:
    def __init__(self, cfg, device):
        from model0 import build_model
        ck = torch.load(ROOT / cfg["model0"]["checkpoint"], map_location=device, weights_only=False)
        self.model = build_model(ck["cfg"]).to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.step = ck["step"]
        self.offsets = context_offsets(ck["cfg"]["data"])
        self.name = f"model0 (step {self.step}, deterministic)"

    @torch.inference_mode()
    def __call__(self, ctx, acts, gens):
        return self.model(ctx, acts).clamp(-1, 1)


# ----------------------------------------------------------------------------- data
def load_episodes(cfg):
    val = set(json.load(open(ROOT / cfg["data"]["val_episodes"]))["val_episode_seeds"])
    eps = []
    for f in episode_files(cfg["data"], ROOT):
        seed = int(f.stem.split("_")[1])
        if seed in val:
            e = np.load(f)
            eps.append({"seed": seed, "native": e["frames"], "actions": e["actions"], "ram": e["ram"]})
    eps.sort(key=lambda e: -len(e["actions"]))
    eps = eps[: cfg["rollout"]["n_episodes"]]
    for e in eps:
        e["frames"] = D.downsample_frames(e["native"], cfg["data"]["size"])
        del e["native"]
    return eps


def analyse_gt(ep, ref, cfg):
    """Per-frame ground-truth detections for one episode (shared across seeds)."""
    T = len(ep["frames"])
    pres = [ref.pellet_presence(f) for f in ep["frames"]]
    det = [ref.sprites(f, p) for f, p in zip(ep["frames"], pres)]
    return {"presence": pres, "det": det, "wall_iou": [ref.wall_iou(f) for f in ep["frames"]],
            "eligible": M.ghost_eligibility(ep["ram"], cfg),
            "dir": M.gt_directions(ep["ram"], cfg), "T": T}


# ----------------------------------------------------------------------------- rollouts
def run_rollouts(cfg, predictor, eps, ref, device, seed):
    r = cfg["rollout"]
    start, H = r["start_step"], r["horizon"]
    jobs = [(ei, s) for ei in range(len(eps)) for s in range(r["seeds"])]
    preds = np.zeros((len(jobs), H, ref.size, ref.size, 3), np.uint8)
    t0 = time.time()
    for b0 in range(0, len(jobs), r["batch"]):
        batch = jobs[b0:b0 + r["batch"]]
        hist = History.from_episodes([(eps[ei]["frames"], eps[ei]["actions"]) for ei, _ in batch], start, predictor.offsets, device)
        gens = [torch.Generator(device=device).manual_seed(seed * 100003 + eps[ei]["seed"] % 100003 + 7919 * s) for ei, s in batch]
        for k in range(H):
            # recorded action that produced frame start + k (wraps once the rollout outlives the episode)
            action = torch.tensor([int(eps[ei]["actions"][(start + k - 1) % len(eps[ei]["actions"])]) for ei, _ in batch], device=device)
            ctx, acts = hist.context(action)
            pred = predictor(ctx, acts, gens)
            preds[b0:b0 + len(batch), k] = to_uint8(pred).permute(0, 2, 3, 1).cpu().numpy()
            hist.push(pred)
            if (k + 1) % 150 == 0:
                print(f"  rollout batch {b0 // r['batch'] + 1}: step {k + 1}/{H} ({time.time() - t0:.0f}s)")
    return jobs, preds


def score(cfg, eps, gts, jobs, preds, ref, model_name):
    r = cfg["rollout"]
    K, start, H = cfg["data"]["context"], r["start_step"], r["horizon"]
    rows = []
    t0 = time.time()
    for j, (ei, s) in enumerate(jobs):
        ep, gt = eps[ei], gts[ei]
        T = gt["T"]
        positions = [gt["det"][start - K + i].get("pac") for i in range(K)]  # context positions
        per_step = []
        for k in range(H):
            i = start + k
            m = preds[j, k]
            pres_m = ref.pellet_presence(m)
            det_m = ref.sprites(m, pres_m)
            positions.append(det_m.get("pac"))
            row = {"model": model_name, "episode": ep["seed"], "seed": s, "step": k + 1, "gt": i < T,
                   "ghost_count_ungated": M.ghost_count(det_m), "wall_iou": ref.wall_iou(m),
                   "pac_detected": det_m.get("pac") is not None, "pellets_remaining": int(pres_m.sum()),
                   "pac_err": np.nan, "ghost_count": np.nan, "ghost_count_gt": np.nan, "ghost_abs_err": np.nan,
                   "pellet_iou": np.nan, "wall_iou_gt": np.nan, "pellets_remaining_gt": np.nan, "resp_event": 0, "resp_hit": 0}
            if i < T:
                err, _, _ = M.pacman_error(det_m, gt["det"][i])
                row["pac_err"] = err
                row["wall_iou_gt"] = gt["wall_iou"][i]
                row["pellets_remaining_gt"] = int(gt["presence"][i].sum())
                row["pellet_iou"] = M.pellet_iou(pres_m, gt["presence"][i])
                if gt["eligible"][i]:
                    row["ghost_count"] = M.ghost_count(det_m)
                    row["ghost_count_gt"] = M.ghost_count(gt["det"][i])
                    row["ghost_abs_err"] = abs(row["ghost_count"] - row["ghost_count_gt"])
            per_step.append(row)
        events = M.responsiveness_events(gt["dir"], ep["actions"], start, min(T, start + H))
        hits = M.responsiveness_hits(events, positions, start, K, cfg)
        for (i, _), h in zip(events, hits):
            per_step[i - start]["resp_event"] = 1
            per_step[i - start]["resp_hit"] = int(h)
        rows.extend(per_step)
        print(f"  scored rollout {j + 1}/{len(jobs)} (episode {ep['seed']}, seed {s}): {len(events)} responsiveness events ({time.time() - t0:.0f}s)")
    return rows


def summarise(cfg, rows, model_name):
    r = cfg["rollout"]
    w = r["window"]
    keys = sorted({(row["episode"], row["seed"]) for row in rows})
    by = {k: [row for row in rows if (row["episode"], row["seed"]) == k] for k in keys}
    out = []

    def add(metric, horizon, gated, values):
        v = np.array([x for x in values if x is not None and not np.isnan(x)], float)
        out.append({"model": model_name, "metric": metric, "horizon": horizon, "gated": gated,
                    "mean": v.mean() if len(v) else np.nan, "std": v.std(ddof=0) if len(v) else np.nan, "n": len(v)})

    for h in r["gated_horizons"]:
        for metric in GATED:
            vals = []
            for k, rs in by.items():
                if metric == "responsiveness":
                    ev = sum(x["resp_event"] for x in rs if x["step"] <= h and x["gt"])
                    vals.append(sum(x["resp_hit"] for x in rs if x["step"] <= h) / ev if ev else np.nan)
                else:
                    win = [x[metric] for x in rs if h - w < x["step"] <= h and x["gt"]]
                    win = [float(x) for x in win if x is not None and not (isinstance(x, float) and np.isnan(x))]
                    vals.append(np.mean(win) if win else np.nan)
            add(metric, h, True, vals)
        ev_total = [sum(x["resp_event"] for x in rs if x["step"] <= h) for rs in by.values()]
        add("responsiveness_events", h, True, ev_total)
    for h in r["ungated_horizons"]:
        for metric in UNGATED:
            vals = []
            for k, rs in by.items():
                win = [float(x[metric]) for x in rs if h - w < x["step"] <= h]
                vals.append(np.mean(win) if win else np.nan)
            add(metric, h, False, vals)
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def plot(result_dirs, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    for d in result_dirs:
        with open(Path(d) / "summary.csv") as f:
            for row in csv.DictReader(f):
                data.setdefault(row["metric"], {}).setdefault(row["model"], []).append(
                    (int(row["horizon"]), float(row["mean"]), float(row["std"]), row["gated"] == "True", int(row["n"])))
    labels = {"pac_err": "Pac-Man position error (px @64x64)", "pac_detected": "Pac-Man detected (fraction)",
              "ghost_count": "ghost count, gated frames (0-4)", "ghost_count_gt": "ghost count, ground truth (0-4)",
              "ghost_abs_err": "|ghost count - GT|", "ghost_count_ungated": "ghost count, ungated (0-4)",
              "pellet_iou": "pellet set IoU", "wall_iou": "wall IoU vs maze", "wall_iou_gt": "wall IoU vs maze, ground truth",
              "pellets_remaining": "pellets remaining", "pellets_remaining_gt": "pellets remaining, ground truth",
              "responsiveness": "action responsiveness (fraction of events)", "responsiveness_events": "responsiveness events (count)"}
    from matplotlib.ticker import NullFormatter, NullLocator
    horizons = sorted({p[0] for per_model in data.values() for pts in per_model.values() for p in pts})
    gated_h = sorted({p[0] for per_model in data.values() for pts in per_model.values() for p in pts if p[3]})
    for metric, per_model in data.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        for mi, (model, pts) in enumerate(sorted(per_model.items())):
            pts.sort()
            g = [p for p in pts if p[3] and p[4] > 0]
            u = [p for p in pts if not p[3] and p[4] > 0]
            if g:
                ax.errorbar([p[0] for p in g], [p[1] for p in g], yerr=[p[2] for p in g], marker="o", capsize=3,
                            label=f"{model} (n={g[0][4]})", color=f"C{mi}")
            if u:
                ax.errorbar([p[0] for p in u], [p[1] for p in u], yerr=[p[2] for p in u], marker="o", mfc="white",
                            capsize=3, linestyle="none", color=f"C{mi}", label=f"{model}, ungated" if not g else None)
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(horizons)
        ax.set_xticklabels([str(h) if h in gated_h else f"{h}*" for h in horizons])
        ax.set_xlabel("rollout horizon (steps; * = ungated, no ground truth; hollow = ungated)")
        ax.set_ylabel(labels.get(metric, metric))
        ax.set_title(labels.get(metric, metric))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}.png", dpi=130)
        plt.close(fig)
    print(f"wrote {len(data)} plots to {out_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--model", choices=["model1", "model0"])
    p.add_argument("--name", help="output subfolder name (default: --model)")
    p.add_argument("--n-episodes", type=int)
    p.add_argument("--seeds", type=int)
    p.add_argument("--horizon", type=int)
    p.add_argument("--plot", nargs="*", help="result dirs to plot together (no rollouts are run)")
    p.add_argument("--save-all-preds", action="store_true", help="save every rollout's frames to preds_all.npy")
    p.add_argument("--no-score", action="store_true", help="run the rollouts and save frames, skip metric scoring")
    a = p.parse_args()
    cfg = load_config(a.config)
    for k in ("n_episodes", "seeds", "horizon"):
        if getattr(a, k) is not None:
            cfg["rollout"][k] = getattr(a, k)
    out_root = ROOT / cfg["out_dir"]
    if a.plot is not None:
        plot(a.plot, out_root / "plots")
        return
    if not a.model:
        raise SystemExit("--model is required unless --plot is given")
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    predictor = (DiffusionPredictor if a.model == "model1" else Model0Predictor)(cfg, device)
    print(f"predictor: {predictor.name} on {device}")
    ref = D.load_reference(cfg)
    eps = load_episodes(cfg)
    print(f"episodes: {[(e['seed'], len(e['actions'])) for e in eps]}")
    gts = [analyse_gt(e, ref, cfg) for e in eps]
    jobs, preds = run_rollouts(cfg, predictor, eps, ref, device, a.seed)
    out_early = out_root / (a.name or a.model)
    out_early.mkdir(parents=True, exist_ok=True)
    if a.save_all_preds:
        np.save(out_early / "preds_all.npy", preds)
        with open(out_early / "jobs.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["row", "episode", "seed", "episode_len"])
            wr.writerows([[j, eps[ei]["seed"], s_, len(eps[ei]["actions"])] for j, (ei, s_) in enumerate(jobs)])
        print(f"saved {out_early}/preds_all.npy {preds.shape} and jobs.csv")
    if a.no_score:
        return
    rows = score(cfg, eps, gts, jobs, preds, ref, a.model)
    summary = summarise(cfg, rows, a.model)
    out = out_root / (a.name or a.model)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "per_step.csv", rows)
    write_csv(out / "summary.csv", summary)
    np.save(out / "preds_first_rollout.npy", preds[0])
    print(f"\n{'metric':24s} " + " ".join(f"{h:>16d}" for h in cfg["rollout"]["gated_horizons"] + cfg["rollout"]["ungated_horizons"]))
    for metric in GATED + ["responsiveness_events"] + [m for m in UNGATED if m not in GATED]:
        cells = []
        for h in cfg["rollout"]["gated_horizons"] + cfg["rollout"]["ungated_horizons"]:
            s = [x for x in summary if x["metric"] == metric and x["horizon"] == h]
            cells.append(f"{s[0]['mean']:7.3f} ± {s[0]['std']:5.3f}" + ("*" if s and not s[0]["gated"] else " ") if s and s[0]["n"] else f"{'-':>16s}")
        print(f"{metric:24s} " + " ".join(f"{c:>16s}" for c in cells))
    print(f"wrote {out}/summary.csv and per_step.csv  (* = ungated)")


if __name__ == "__main__":
    main()
