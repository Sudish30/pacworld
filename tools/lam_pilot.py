"""LAM v2 pilot gate (notes/lam_v2_design.md section 2). Frames only: no actions, no RAM.

Reads <checkpoint_dir>/lam.pt, which must be at step pilot.steps, and scores the three pre-registered conditions:
  1. perplexity >= pilot.min_perplexity          (the fixed val batches of training, as train_lam.py's eval)
  2. code gain   > pilot.min_code_gain           (shuffled / inferred val MSE, same batches)
  3. Pac-Man share > ghost share of the attribution split:
       pilot.attribution_contexts fixed val contexts; each is decoded with all n_codes codes; spread = the variance
       of the n outputs per pixel, averaged over RGB. A pixel is Pac-Man's if within pilot.attribution_radius_px of
       his detected centre in the last context frame, a ghost's if within that distance of any detected ghost
       (four colours and every frightened blob). Pixels near both are a separate "both" bucket and count for
       neither. share = spread summed over the bucket and over all contexts / total spread.
All three must hold to continue the run; otherwise stop and report. Writes <checkpoint_dir>/pilot.json.

  python tools/lam_pilot.py --config configs/lam-v2.yaml --seed 0 [--mechanics]
      --mechanics: Mac smoke runs only; skips the step check and marks the output as not a pilot result.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from common import load_config  # noqa: E402
from dataset import FrameCodec, codec_of, get_datasets  # noqa: E402
from lam import LAM  # noqa: E402
from train_lam import eval_lam, to_dev  # noqa: E402
from train_model0 import pick_device  # noqa: E402
import detectors as D  # noqa: E402


def near(centres, size, radius):
    rr = np.arange(size, dtype=np.float32)
    m = np.zeros((size, size), bool)
    for r, c in centres:
        m |= (rr[:, None] - r) ** 2 + (rr[None, :] - c) ** 2 <= radius ** 2
    return m


@torch.no_grad()
def attribution(model, val, cfg, cache, codec, device):
    p, n, size = cfg["pilot"], cfg["lam"]["n_codes"], cfg["data"]["size"]
    ref, np_codec = D.load_reference(cfg), codec_of(cache)
    targets = val.fixed_batches(p["attribution_contexts"], 1, seed=p["attribution_seed"])[0]
    tot = {"pac": 0.0, "ghost": 0.0, "both": 0.0, "elsewhere": 0.0}
    area = dict(tot)
    found = {"pac": 0, "ghost": 0}
    bs = cfg["train"]["batch_size"]
    for b0 in range(0, len(targets), bs):
        ctx_u8, _, _ = val.get_raw(targets[b0:b0 + bs])
        ctx = to_dev(codec, ctx_u8, device)
        outs = torch.stack([model.decode(ctx, model.vq.embed[k].expand(len(ctx), -1)) for k in range(n)])
        spread = outs.var(0, unbiased=False).mean(1).cpu().numpy()            # (B, H, W)
        last = np_codec.decode_np(ctx_u8[:, -1].numpy())
        for k in range(len(ctx)):
            sp = ref.sprites(last[k])
            pac = [sp["pac"][:2]] if sp["pac"] is not None else []
            ghosts = [sp[g][:2] for g in D.GHOST_NAMES if sp[g] is not None] + [b[:2] for b in ref.frightened_blobs(last[k])]
            mp, mg = near(pac, size, p["attribution_radius_px"]), near(ghosts, size, p["attribution_radius_px"])
            found["pac"] += bool(pac)
            found["ghost"] += bool(ghosts)
            for name, m in (("pac", mp & ~mg), ("ghost", mg & ~mp), ("both", mp & mg), ("elsewhere", ~mp & ~mg)):
                tot[name] += float(spread[k][m].sum())
                area[name] += int(m.sum())
    total = sum(tot.values())
    return {"contexts": int(len(targets)), "contexts_with_pac": found["pac"], "contexts_with_ghost": found["ghost"],
            "share": {k: (v / total if total > 0 else 0.0) for k, v in tot.items()}, "total_spread": total,
            "mean_spread_per_pixel": {k: (tot[k] / area[k] if area[k] else 0.0) for k in tot}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mechanics", action="store_true")
    ap.add_argument("--batch-size", type=int, help="with --mechanics only: the batch size of the smoke run")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if a.batch_size and not a.mechanics:
        raise SystemExit("--batch-size is for --mechanics runs only")
    cfg = load_config(ROOT / a.config)
    p, tr = cfg["pilot"], cfg["train"]
    device = pick_device(tr["device"])
    ck = torch.load(ROOT / tr["checkpoint_dir"] / "lam.pt", map_location=device)
    if ck["step"] != p["steps"] and not a.mechanics:
        raise SystemExit(f"the checkpoint is at step {ck['step']}, the pilot is scored at step {p['steps']}")
    _, val, cache, _ = get_datasets(cfg)
    codec = FrameCodec(cache.get("palette")).to(device)
    model = LAM(ck["cfg"]).to(device).eval()
    model.load_state_dict(ck["model"])
    pac = torch.from_numpy(np.load(ROOT / cfg["data"]["pac_positions"])) if cfg["lam"]["loss_weight"]["enabled"] else None
    bs = a.batch_size or tr["batch_size"]
    ev, _ = eval_lam(model, val, cfg, codec, pac, device, val.fixed_batches(bs, tr["eval_batches"], seed=a.seed),
                     tr["bf16"] and device.type == "cuda")
    model.eval()
    att = attribution(model, val, cfg, cache, codec, device)
    conds = {"perplexity": bool(ev["perplexity"] >= p["min_perplexity"]),
             "code_gain": bool(ev["code_gain"] > p["min_code_gain"]),
             "pac_share_gt_ghost_share": bool(att["share"]["pac"] > att["share"]["ghost"])}
    res = {"step": ck["step"], "mechanics_only": a.mechanics, "perplexity": ev["perplexity"], "code_gain": ev["code_gain"],
           "mse": ev["mse"], "mse_shuffled": ev["mse_shuffled"], "usage": ev["usage"],
           "entropy_weight": float(model.vq.entropy_weight) if model.vq.entropy else None,
           "attribution": att, "conditions": conds, "pass": all(conds.values())}
    out = ROOT / tr["checkpoint_dir"] / "pilot.json"
    json.dump(res, open(out, "w"), indent=2)
    sh = att["share"]
    print(f"step {ck['step']}: perplexity {ev['perplexity']:.2f} (>= {p['min_perplexity']}: {conds['perplexity']}) | "
          f"code gain x{ev['code_gain']:.4f} (> {p['min_code_gain']}: {conds['code_gain']}) | spread share pac {sh['pac']:.3f} "
          f"ghost {sh['ghost']:.3f} both {sh['both']:.3f} elsewhere {sh['elsewhere']:.3f} (pac > ghost: {conds['pac_share_gt_ghost_share']})")
    print(("MECHANICS ONLY, not a pilot result" if a.mechanics else ("PILOT PASSES: continue" if res["pass"] else "PILOT FAILS: stop and report")) + f" -> {out}")


if __name__ == "__main__":
    main()
