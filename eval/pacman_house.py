"""Hidden-timer rule part b: the 2600 Pac-Man ghost-house stay (notes/timer_rule_design.md section 9 + Amendment 1).

Truth: at native resolution the house is occupied when the box holds >= native_min_px ghost-pink pixels. A start is an
occupancy onset c whose real occupancy lasts >= min_stay steps and that has >= min_history real frames before it. The
truth lag is the step, counted from c, of the first of two consecutive empty frames (35 or 36 in the sample).

  --ideal-only  P-D0 (the 64px detector reproduces the truth lag on every val start) and P-I0 (the ideal observer
                limited to the layout's offsets: house-occupied bits at the offsets, hazard from the train episodes)
  default       also rolls out the layout's model from the real context up to and including c, recorded actions,
                3-step Euler, context sigma 0; lag from the 64px detector; parked = no release within the horizon.
Metrics: on-time fraction (|lag - truth| <= tol), median |lag - truth| (parked = horizon), release fraction; bootstrap
95% CIs over val episodes.

  python eval/pacman_house.py --config configs/pacman_house.yaml --layout S10 --seed 0 [--ideal-only]
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
from dataset import History, episode_files, resize_frames, to_uint8  # noqa: E402


def occ_native(frames, h):
    r0, r1 = h["native_rows"]
    c0, c1 = h["native_cols"]
    d = np.abs(frames[:, r0:r1, c0:c1].astype(int) - np.array(h["ghost_rgb"])).sum(-1)
    return (d < h["ghost_tol"]).sum((1, 2)) >= h["native_min_px"]


def occ_small(small, ref, h):
    r0, r1 = h["small_rows"]
    c0, c1 = h["small_cols"]
    return (np.linalg.norm(small[:, r0:r1, c0:c1].astype(float) - ref, axis=-1) > h["small_resid"]).sum((1, 2)) >= h["small_min_px"]


def release_lag(occ, c, horizon):
    """Steps from c to the first of two consecutive empty frames (None = still occupied within the horizon)."""
    for t in range(c + 1, min(len(occ) - 1, c + horizon + 1)):
        if not occ[t] and not occ[t + 1]:
            return t - c
    return None


def split_episodes(cfg):
    val = set(json.load(open(ROOT / cfg["data"]["val_episodes"]))["val_episode_seeds"])
    files = episode_files(cfg["data"], ROOT)
    return [f for f in files if int(f.stem.split("_")[1]) not in val], [f for f in files if int(f.stem.split("_")[1]) in val]


def starts_of(occ, s):
    out = []
    for c in range(1, len(occ)):
        if occ[c] and not occ[c - 1] and c >= s["min_history"] and occ[c:c + s["min_stay"]].all():
            out.append(c)
    return out


def pattern(bits, idx, offsets):
    return tuple(bool(bits[max(idx + o, 0)]) for o in offsets)


def metrics(lags, truths, ep_of, horizon, tol, B, seed):
    lags, truths, ep_of = np.asarray(lags), np.asarray(truths), np.asarray(ep_of)
    err = np.where(lags > 0, np.abs(lags - truths), horizon - truths).astype(float)

    def stats(ix):
        l, e, tr = lags[ix], err[ix], truths[ix]
        return {"on_time": float(np.mean((l > 0) & (np.abs(l - tr) <= tol))), "median_abs_err": float(np.median(e)),
                "release_frac": float(np.mean(l > 0)), "early_frac": float(np.mean((l > 0) & (l < tr - tol)))}

    point = stats(np.arange(len(lags)))
    eps = np.unique(ep_of)
    rng = np.random.default_rng(seed)
    boots = [stats(np.concatenate([np.nonzero(ep_of == p)[0] for p in rng.choice(eps, len(eps))])) for _ in range(B)]
    ci = {k: [float(np.percentile([b[k] for b in boots], 2.5)), float(np.percentile([b[k] for b in boots], 97.5))] for k in point}
    return {"point": point, "ci95": ci, "n": int(len(lags)), "episodes": int(len(eps))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/pacman_house.yaml")
    p.add_argument("--layout", required=True, choices=["C10", "S10"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--ideal-only", action="store_true")
    p.add_argument("--checkpoint", help="override the layout's checkpoint (pilots)")
    p.add_argument("--max-val-episodes", type=int, help="evaluate on the first N val episodes only (pilots)")
    a = p.parse_args()
    cfg = load_config(ROOT / a.config)
    h, s, e = cfg["house"], cfg["starts"], cfg["eval"]
    lay = cfg["layouts"][a.layout]
    offsets = lay["offsets"]
    t0 = time.time()
    train_f, val_f = split_episodes(cfg)
    if a.max_val_episodes:
        val_f = val_f[: a.max_val_episodes]

    # empty-house reference at 64px and the ideal observer's hazard, from TRAIN episodes only
    refs, n, r = [], {}, {}
    for f in train_f:
        fr = np.load(f)["frames"]
        occ = occ_native(fr, h)
        small = resize_frames(fr[~occ][::7], cfg["data"]["size"], cfg["data"]["resample"])
        refs.append(small[:, slice(*h["small_rows"]), slice(*h["small_cols"])])
        for t in range(1, len(occ)):
            if not occ[t - 1]:
                continue
            k = pattern(occ, t, offsets)
            n[k] = n.get(k, 0) + 1
            if not occ[t]:
                r[k] = r.get(k, 0) + 1
    ref = np.median(np.concatenate(refs), 0).astype(float)
    hazard = {k: r.get(k, 0) / n[k] for k in n}

    # val starts, truth, P-D0 and the ideal observer
    rng = np.random.default_rng(a.seed)
    starts, truths, d0, obs, obs_truth, obs_ep, unseen = [], [], [], [], [], [], 0
    val_eps = {}
    for f in val_f:
        ep = np.load(f)
        fr, acts = ep["frames"], ep["actions"]
        occ = occ_native(fr, h)
        small = resize_frames(fr, cfg["data"]["size"], cfg["data"]["resample"])
        o64 = occ_small(small, ref, h)
        seed_ = int(f.stem.split("_")[1])
        val_eps[seed_] = (small, acts)
        for c in starts_of(occ, s):
            tl = release_lag(occ, c, e["horizon"])
            if tl is None:
                continue
            starts.append((seed_, c))
            truths.append(tl)
            d0.append(release_lag(o64, c, e["horizon"]) == tl)
            seq = np.ones(c + 1 + e["horizon"], bool)
            seq[: c + 1] = occ[: c + 1]
            haz = np.array([hazard.get(pattern(seq, c + k, offsets), 0.0) for k in range(1, e["horizon"] + 1)])
            unseen += int(sum(pattern(seq, c + k, offsets) not in hazard for k in range(1, e["horizon"] + 1)))
            fire = rng.random((e["ideal_draws"], e["horizon"])) < haz[None]
            lg = np.where(fire.any(1), fire.argmax(1) + 1, -1)
            obs.append(lg)
            obs_truth += [tl] * len(lg)
            obs_ep += [seed_] * len(lg)
    ep_of = [sd for sd, _ in starts]
    res = {"layout": a.layout, "offsets": offsets, "val_episodes": len(val_f), "train_episodes": len(train_f), "starts": len(starts),
           "truth_lags": {int(k): int(v) for k, v in zip(*np.unique(truths, return_counts=True))},
           "P-D0": {"pass": bool(all(d0)), "agree": int(sum(d0)), "of": len(d0)},
           "ideal": metrics(np.concatenate(obs), obs_truth, obs_ep, e["horizon"], e["tol"], e["bootstrap"], a.seed),
           "ideal_unseen_pattern_steps": unseen, "ideal_patterns": len(hazard)}

    if not a.ideal_only:
        from model1 import build_model, euler_sample
        from train_model0 import pick_device
        device = pick_device("auto")
        ck = torch.load(ROOT / (a.checkpoint or lay["checkpoint"]), map_location=device)
        model = build_model(ck["cfg"]).to(device).eval()
        model.load_state_dict(ck["ema"])
        if list(ck["cfg"]["data"]["context_offsets"]) != list(offsets):
            raise SystemExit("checkpoint context offsets differ from the layout")
        lags = np.full(len(starts), -1, np.int64)
        reach = -min(offsets)
        with torch.no_grad():
            for b0 in range(0, len(starts), e["batch"]):
                batch = starts[b0:b0 + e["batch"]]
                eps = [(val_eps[sd][0][c - reach:c + 1], val_eps[sd][1][c - reach:c + 1]) for sd, c in batch]
                hist = History.from_episodes(eps, reach + 1, offsets, device)
                gen = torch.Generator(device=device).manual_seed(a.seed * 100_003 + b0)
                done = np.zeros(len(batch), bool)
                empty_prev = np.zeros(len(batch), bool)
                sig = torch.full((len(batch),), float(e["ctx_sigma"]), device=device)
                for k in range(1, e["horizon"] + 2):
                    act = torch.tensor([int(val_eps[sd][1][min(c + k - 1, len(val_eps[sd][1]) - 1)]) for sd, c in batch], device=device)
                    ctx, acts = hist.context(act)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        pred = euler_sample(model, ctx, acts, sig, e["sampler_steps"], ck["cfg"]["diffusion"], gen).float()
                    hist.push(pred)
                    empty = ~occ_small(to_uint8(pred).permute(0, 2, 3, 1).cpu().numpy(), ref, h)
                    new = empty & empty_prev & ~done & (k - 1 <= e["horizon"])      # two consecutive empty frames: k-1, k
                    lags[b0 + np.nonzero(new)[0]] = k - 1
                    done |= new
                    empty_prev = empty
                    if done.all():
                        break
        res["model"] = metrics(lags, truths, ep_of, e["horizon"], e["tol"], e["bootstrap"], a.seed)
        res["model"]["lags"] = lags.tolist()
        res["model_checkpoint"] = a.checkpoint or lay["checkpoint"]
        res["model_step"] = ck["step"]
    out = ROOT / e["out_dir"] / f"{a.layout}{'-ideal' if a.ideal_only else ''}{'-pilot' if a.checkpoint else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    show = lambda m: {k: round(v, 3) for k, v in m["point"].items()}
    print(f"{a.layout}: {len(starts)} starts in {len(val_f)} val episodes; truth lags {res['truth_lags']}; "
          f"P-D0 {res['P-D0']['agree']}/{res['P-D0']['of']}; ideal {show(res['ideal'])}"
          + (f"; model {show(res['model'])}" if "model" in res else "") + f"  ({time.time() - t0:.0f}s) -> {out}")


if __name__ == "__main__":
    main()
