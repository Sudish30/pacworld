"""Score one cell (layout, N) of the hidden-timer experiment (notes/timer_rule_design.md).

  --ideal-only  D0 (the detector on real test frames) and I0 (the ideal observer limited to the layout's offsets,
                estimated from the training episodes) - no model needed; run for every cell before any model trains.
  default       also rolls out the trained model from checkpoints/timer/<layout>-N<N>/model.pt.

Rollouts start at captures in the test episodes (real frames up to and including the capture frame c), 3-step
Euler, context sigma 0, horizon 2N + 50. Lag = first frame after c the detector reads as released; none = parked.
Metrics (model and ideal observer, same starts): on-time fraction (tol_strict and tol_rel), median |lag - N| (parked
= horizon), release fraction, early-release fraction; bootstrap 95% CIs over test episodes.

  python eval/timer_eval.py --config configs/timer.yaml --layout S10 --N 33 --seed 0 [--ideal-only]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import load_config  # noqa: E402
from dataset import History, to_uint8  # noqa: E402
from timer_game import (generate_split, ghost_state, observer_hazard, observer_rollouts, release_lag,  # noqa: E402
                        rollout_starts)


def metrics(lags, ep_of, N, horizon, tols, B, seed):
    """lags (S,) with -1 = parked, grouped by test episode ep_of (S,). Point estimates and episode-bootstrap CIs."""
    lags, ep_of = np.asarray(lags), np.asarray(ep_of)
    err = np.where(lags > 0, np.abs(lags - N), horizon - N).astype(float)

    def stats(ix):
        l, e = lags[ix], err[ix]
        out = {f"on_time_tol{t:g}": float(np.mean((l > 0) & (np.abs(l - N) <= t))) for t in tols}
        out.update(median_abs_err=float(np.median(e)), release_frac=float(np.mean(l > 0)),
                   early_frac=float(np.mean((l > 0) & (l < N - tols[-1]))))
        return out

    point = stats(np.arange(len(lags)))
    eps = np.unique(ep_of)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        pick = rng.choice(eps, len(eps), replace=True)
        ix = np.concatenate([np.nonzero(ep_of == p)[0] for p in pick])
        boots.append(stats(ix))
    ci = {k: [float(np.percentile([b[k] for b in boots], 2.5)), float(np.percentile([b[k] for b in boots], 97.5))] for k in point}
    return {"point": point, "ci95": ci, "n": int(len(lags)), "episodes": int(len(eps))}


@torch.no_grad()
def model_lags(ck_path, test, starts, horizon, offsets, cfg, device, seed):
    from model1 import build_model, euler_sample
    ck = torch.load(ck_path, map_location=device)
    mcfg = ck["cfg"]
    model = build_model(mcfg).to(device).eval()
    model.load_state_dict(ck["ema"])
    e, d = cfg["eval"], mcfg["diffusion"]
    g = cfg["game"]
    reach = -min(offsets)
    lags = np.full(len(starts), -1, np.int64)
    use_bf16 = device.type == "cuda" and mcfg["train"]["bf16"]
    for b0 in range(0, len(starts), e["batch"]):
        batch = starts[b0:b0 + e["batch"]]
        # History from the last reach+1 real frames up to and including the capture frame c (first prediction: c+1)
        eps =[(test[i][0][c + 1 - min(c + 1, reach + 1):c + 1], np.zeros(min(c + 1, reach + 1), np.int64)) for i, c in batch]
        hist = History.from_episodes(eps, len(eps[0][0]), offsets, device)
        gen = torch.Generator(device=device).manual_seed(seed * 100_003 + b0)
        done = np.zeros(len(batch), bool)
        zero = torch.zeros(len(batch), dtype=torch.long, device=device)
        sig = torch.full((len(batch),), float(e["ctx_sigma"]), device=device)
        for k in range(1, horizon + 1):
            ctx, acts = hist.context(zero)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                pred = euler_sample(model, ctx, acts, sig, e["sampler_steps"], d, gen).float()
            hist.push(pred)
            _, rel = ghost_state(to_uint8(pred).permute(0, 2, 3, 1).cpu().numpy(), g)
            new = rel & ~done
            lags[b0 + np.nonzero(new)[0]] = k
            done |= rel
            if done.all():
                break
    return lags, ck["step"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/timer.yaml")
    p.add_argument("--layout", required=True)
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--ideal-only", action="store_true")
    p.add_argument("--checkpoint", help="default <checkpoint_root>/<layout>-N<N>/model.pt")
    a = p.parse_args()
    cfg = load_config(ROOT / a.config)
    g, e = cfg["game"], cfg["eval"]
    cells = cfg["grid"].get(a.layout, []) + cfg.get("followup_grid", {}).get(a.layout, [])
    if a.N not in cells:
        raise SystemExit(f"N={a.N} is not a pre-registered cell of {a.layout}: {cells}")
    offsets = cfg["layouts"][a.layout]
    N = a.N
    t0 = time.time()
    test, test_sha = generate_split(g, N, "test")
    starts, horizon = rollout_starts(test, N, e)
    ep_of = [i for i, _ in starts]
    tols = [e["tol_strict"], max(e["tol_strict"], e["tol_rel"] * N)]

    # D0: the detector on the real test frames gives lag = N for every start
    real = [release_lag(ghost_state(test[i][0][c + 1:c + 1 + horizon], g)[1]) for i, c in starts]
    d0 = all(l == N for l in real)

    # I0: the ideal observer limited to these offsets, hazard from the training episodes
    train, train_sha = generate_split(g, N, "train", render=False)
    hazard = observer_hazard(train, offsets)
    rng = np.random.default_rng(a.seed)
    obs, obs_ep, unseen = [], [], 0
    for (i, c) in starts:
        l, u = observer_rollouts(hazard, test[i][1], c, horizon, offsets, e["ideal_draws"], rng)
        obs.append(l)
        obs_ep += [i] * len(l)
        unseen += u
    res = {"layout": a.layout, "N": N, "reach": -min(offsets), "offsets": offsets, "horizon": horizon, "tols": tols,
           "starts": len(starts), "test_sha256": test_sha, "train_sha256_penned": train_sha,
           "D0_detector_pass": d0, "real_lags": sorted(set(real), key=lambda x: (x is None, x)),
           "ideal": metrics(np.concatenate(obs), obs_ep, N, horizon, tols, e["bootstrap"], a.seed),
           "ideal_unseen_pattern_steps": int(unseen), "ideal_patterns": len(hazard)}
    if not a.ideal_only:
        from train_model0 import pick_device
        device = pick_device(cfg["train"]["device"])
        ck = Path(a.checkpoint or ROOT / cfg["train"]["checkpoint_root"] / f"{a.layout}-N{N}" / "model.pt")
        lags, step = model_lags(ck, test, starts, horizon, offsets, cfg, device, a.seed)
        res["model"] = metrics(lags, ep_of, N, horizon, tols, e["bootstrap"], a.seed)
        res["model"]["lags"] = lags.tolist()
        res["model_checkpoint"], res["model_step"] = str(ck), step
    out = ROOT / e["out_dir"] / f"{a.layout}-N{N}{'-ideal' if a.ideal_only else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    show = lambda m: {k: round(v, 3) for k, v in m["point"].items()}
    print(f"{a.layout} N={N}: D0 {'pass' if d0 else 'FAIL'}; ideal {show(res['ideal'])}"
          + (f"; model {show(res['model'])}" if "model" in res else "") + f"  ({time.time() - t0:.0f}s) -> {out}")


if __name__ == "__main__":
    main()
