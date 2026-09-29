"""Stage 1 label-free checks L1-L3 and T_det for one LAM arm, and the pre-registered arm choice (notes/handoff.md).
Frames only: no actions, no RAM. Run after tools/lam_codes.py; it needs no prior.

  L1    the label-free key map (tools/lam_keymap.code_directions on TRAIN transitions) gives every direction a code
  L2    codebook perplexity over all VAL transitions >= checks.l2_min_perplexity
  L3    code gain on VAL: decoder MSE with codes shuffled within the batch / with the inferred codes; paired bootstrap
        over samples (checks.bootstrap resamples); pass if the ratio > 1 and its 95% CI excludes 1
  T_det on VAL transitions where the pixel detector sees Pac-Man's movement direction change (both moves >= min move),
        accuracy of dir(code of the new move's transition) at the new direction

  python tools/lam_checks.py --config configs/lam-A.yaml --seed 0         -> <checkpoint_dir>/label_free_checks.json
  python tools/lam_checks.py --select configs/lam-A.yaml configs/lam-B.yaml --seed 0
      the choice: among arms passing L1-L3 the higher T_det wins; A wins unless B is higher by more than
      checks.select_margin. Writes checks.selection_file. No arm left = STOP (labels are never read).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from common import load_config  # noqa: E402
from dataset import FrameCodec, get_datasets  # noqa: E402
from lam import LAM, encoder_frames, perplexity  # noqa: E402
from lam_keymap import DIRECTIONS, code_directions, move_directions, train_transitions  # noqa: E402
from train_model0 import pick_device  # noqa: E402


def episode_transitions(cache, idx):
    s = cache["ep_start"]
    return np.concatenate([np.arange(s[e], s[e + 1] - 1) for e in idx])


@torch.no_grad()
def code_gain(cfg, val, cache, device, seed):
    """Per-sample decoder MSE with inferred codes and with codes shuffled within each batch (fixed permutations)."""
    ck = torch.load(ROOT / cfg["train"]["checkpoint_dir"] / "lam.pt", map_location=device)
    model = LAM(ck["cfg"]).to(device).eval()
    model.load_state_dict(ck["model"])
    codec = FrameCodec(cache.get("palette")).to(device)
    dec = lambda u8: codec.decode(u8.to(device) if u8.dim() >= 4 + (codec.palette is None) else u8.to(device)[:, None])
    g = torch.Generator().manual_seed(seed)
    inferred, shuffled = [], []
    for t in val.fixed_batches(cfg["train"]["batch_size"], cfg["checks"]["l3_batches"], seed=seed):
        ctx_u8, _, tgt_u8 = val.get_raw(t)
        ctx, tgt = dec(ctx_u8), dec(tgt_u8)
        enc = dec(encoder_frames(val, t, cfg["lam"]["encoder_offsets"]))
        pred, _, _, q = model(enc, ctx)
        pred_sh = model.decode(ctx, q[torch.randperm(len(t), generator=g).to(device)])
        inferred.append(((pred - tgt) ** 2).mean((1, 2, 3)).cpu())
        shuffled.append(((pred_sh - tgt) ** 2).mean((1, 2, 3)).cpu())
    return torch.cat(inferred).numpy(), torch.cat(shuffled).numpy(), ck["step"]


def run_checks(cfg, seed):
    c, n = cfg["checks"], cfg["lam"]["n_codes"]
    cfg["data"]["action_source"] = "lam"                     # codes in the action slots; the labels are deleted
    _, val, cache, (train_idx, val_idx) = get_datasets(cfg)
    codes, pac = cache["actions"], np.load(ROOT / cfg["data"]["pac_positions"])
    dirs, table = code_directions(codes, pac, train_transitions(cache, train_idx), n, cfg["keymap"])
    missing = [d for d in DIRECTIONS if d not in dirs]
    vt = episode_transitions(cache, val_idx)
    ppl = perplexity(torch.bincount(torch.from_numpy(codes[vt]), minlength=n))

    device = pick_device(cfg["train"]["device"])
    mse_i, mse_s, lam_step = code_gain(cfg, val, cache, device, seed)
    rng = np.random.default_rng(seed)
    boot = [mse_s[k].mean() / mse_i[k].mean() for k in (rng.integers(0, len(mse_i), len(mse_i)) for _ in range(c["bootstrap"]))]
    ratio, lo, hi = float(mse_s.mean() / mse_i.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    # T_det: turns the pixel detector sees; j is the transition of the new move, j - 1 the one before
    j = vt[np.isin(vt - 1, vt)]
    k = cfg["keymap"]
    before = move_directions(pac, j - 1, c["tdet_min_move_px"], k["max_move_px"])
    after = move_directions(pac, j, c["tdet_min_move_px"], k["max_move_px"])
    turn = (before >= 0) & (after >= 0) & (before != after)
    pred = np.array([DIRECTIONS.index(dirs[x]) if dirs[x] != "NONE" else -9 for x in codes[j[turn]]])
    t_det = float((pred == after[turn]).mean()) if turn.any() else 0.0

    res = {"lam_step": lam_step, "dir": dirs, "table": table,
           "L1": {"pass": not missing, "missing": missing},
           "L2": {"pass": ppl >= c["l2_min_perplexity"], "perplexity": ppl, "min": c["l2_min_perplexity"]},
           "L3": {"pass": ratio > 1 and lo > 1, "ratio": ratio, "ci95": [lo, hi], "samples": int(len(mse_i)),
                  "mse_inferred": float(mse_i.mean()), "mse_shuffled": float(mse_s.mean())},
           "T_det": {"accuracy": t_det, "detector_turns": int(turn.sum()), "val_transitions": int(len(vt))}}
    res["passes_L1_L3"] = res["L1"]["pass"] and res["L2"]["pass"] and res["L3"]["pass"]
    return res


def select(paths, seed):
    arms = {}
    for p in paths:
        cfg = load_config(ROOT / p)
        f = ROOT / cfg["train"]["checkpoint_dir"] / "label_free_checks.json"
        arms[Path(p).stem] = (json.load(open(f)), cfg["checks"]["select_margin"])
        sel_file = ROOT / cfg["checks"]["selection_file"]
    ok = {a: r for a, (r, _) in arms.items() if r["passes_L1_L3"]}
    margin = next(iter(arms.values()))[1]
    if not ok:
        choice, why = None, "no arm passes L1-L3: STOP (labels are never read)"
    elif len(ok) == 1:
        choice = next(iter(ok))
        why = "the only arm passing L1-L3"
    else:
        a = next(x for x in ok if x.startswith("lam-A"))
        b = next(x for x in ok if x.startswith("lam-B"))
        ta, tb = ok[a]["T_det"]["accuracy"], ok[b]["T_det"]["accuracy"]
        choice = b if tb > ta + margin else a
        why = f"T_det A {ta:.4f} vs B {tb:.4f}; B wins only if higher by more than {margin}"
    out = {"choice": choice, "reason": why, "arms": {a: {"passes_L1_L3": r["passes_L1_L3"], "T_det": r["T_det"]["accuracy"],
                                                            "L2_perplexity": r["L2"]["perplexity"], "L3_ratio": r["L3"]["ratio"]}
                                                     for a, (r, _) in arms.items()}}
    json.dump(out, open(sel_file, "w"), indent=2)
    print(json.dumps(out, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("--select", nargs="+", metavar="CONFIG")
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    if a.select:
        return select(a.select, a.seed)
    cfg = load_config(ROOT / a.config)
    res = run_checks(cfg, a.seed)
    path = ROOT / cfg["train"]["checkpoint_dir"] / "label_free_checks.json"
    json.dump(res, open(path, "w"), indent=2)
    for row in res["table"]:
        print(f"  code {row['code']}: {row['dir']:5s} purity {row['purity']:.2f} share {row['share']:.3f}")
    print(f"L1 {'pass' if res['L1']['pass'] else 'FAIL ' + str(res['L1']['missing'])} | "
          f"L2 perplexity {res['L2']['perplexity']:.2f} {'pass' if res['L2']['pass'] else 'FAIL'} | "
          f"L3 ratio {res['L3']['ratio']:.4f} CI [{res['L3']['ci95'][0]:.4f}, {res['L3']['ci95'][1]:.4f}] {'pass' if res['L3']['pass'] else 'FAIL'} | "
          f"T_det {res['T_det']['accuracy']:.4f} on {res['T_det']['detector_turns']} detector turns; wrote {path}")


if __name__ == "__main__":
    main()
