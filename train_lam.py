"""Train the latent action model (--part lam) or its prior (--part prior). Frames only: no actions, no RAM.

--part lam    encoder + VQ + decoder on frames (data.action_source must be 'none'). Every train.eval_every steps, on
              fixed val batches: reconstruction MSE with the inferred codes and with codes shuffled within the batch
              (code gain), codebook perplexity and usage, and a code-swap grid (outputs/<run>/swap_<step>.png: each
              val context decoded with every code). Checkpoint: <checkpoint_dir>/lam.pt (final weights are used).
--part prior  p(code of the next transition | the world model's context), trained with cross-entropy on data.lam_codes
              (written by tools/lam_codes.py from the finished LAM). Checkpoint: <checkpoint_dir>/prior.pt.

  python train_lam.py --config configs/lam-A.yaml --seed 0 --part lam
  python train_lam.py --config configs/lam-A.yaml --seed 0 --part prior

Label-firewall test (tools/firewall_test.py): --firewall-scramble-seed replaces the recorded actions by random integers
before anything reads them, --dump saves every loss, the final val codes (LAM: inferred codes; prior: predicted codes) and a weights hash; the
two runs must match.
"""
import argparse
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from common import load_config
from dataset import FrameCodec, get_datasets, to_uint8
from lam import LAM, Prior, cosine_lr, encoder_frames, perplexity, player_weight, weighted_mse
from train_model0 import pick_device, upscale


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--part", choices=["lam", "prior"], required=True)
    p.add_argument("--steps", type=int, help="override the step count (smoke tests)")
    p.add_argument("--batch-size", type=int, help="override the batch size (smoke tests)")
    p.add_argument("--eval-every", type=int, help="override eval_every (smoke tests)")
    p.add_argument("--resume", action="store_true", help="continue from this part's checkpoint")
    p.add_argument("--stop-at", type=int, help="stop (eval + checkpoint) at this step; the LR schedule still spans "
                                               "train.steps, so --resume continues the same run (LAM v2 pilot)")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--run-name")
    p.add_argument("--firewall-scramble-seed", type=int, help="label-firewall test only (see the module docstring)")
    p.add_argument("--dump", help="label-firewall test only: save losses, final val codes and a weights hash here")
    return p.parse_args()


def weights_sha256(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


class Batches:
    """Draws LAM or prior batches as raw cached bytes (decoded on the device), optionally one step ahead in a thread."""

    def __init__(self, ds, cfg, part, pac, batch_size, generator, prefetch):
        self.ds, self.part, self.pac, self.bs, self.gen = ds, part, pac, batch_size, generator
        self.enc_offsets = cfg["lam"]["encoder_offsets"]
        self.pool = ThreadPoolExecutor(1) if prefetch else None
        self.pending = self.pool.submit(self._draw) if prefetch else None

    def get(self, targets):
        ctx, acts, tgt = self.ds.get_raw(targets)
        if self.part == "prior":
            return ctx, acts[:, -1]                       # in lam mode the last action slot holds codes[i-1]: the label
        enc = encoder_frames(self.ds, targets, self.enc_offsets)
        pac = self.pac[torch.stack([targets - 1, targets], 1)] if self.pac is not None else torch.full((len(targets), 2, 2), float("nan"))
        return ctx, enc, tgt, pac

    def _draw(self):
        pick = torch.randint(len(self.ds.targets), (self.bs,), generator=self.gen)
        return self.get(self.ds.targets[pick])

    def next(self):
        if self.pool is None:
            return self._draw()
        b = self.pending.result()
        self.pending = self.pool.submit(self._draw)
        return b


def to_dev(codec, u8, device):
    """(B, K, H, W[, 3]) or (B, H, W[, 3]) cached bytes -> (B, 3K, H, W) float in [-1, 1] on device."""
    u8 = u8.to(device, non_blocking=True)
    return codec.decode(u8 if u8.dim() >= 4 + (codec.palette is None) else u8[:, None])


# ----------------------------------------------------------------------------- LAM evaluation
@torch.no_grad()
def eval_lam(model, val, cfg, codec, pac, device, batches, use_bf16):
    model.eval()
    lw, n = cfg["lam"]["loss_weight"], cfg["lam"]["n_codes"]
    B = Batches(val, cfg, "lam", pac, 1, None, False)
    tot = {"mse": 0.0, "weighted": 0.0, "mse_shuffled": 0.0}
    counts = torch.zeros(n, dtype=torch.long)
    codes = []
    g = torch.Generator().manual_seed(0)
    for t in batches:
        ctx_u8, enc_u8, tgt_u8, pac_b = B.get(t)
        ctx, enc, tgt = to_dev(codec, ctx_u8, device), to_dev(codec, enc_u8, device), to_dev(codec, tgt_u8, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            pred, idx, _, q = model(enc, ctx)
            perm = torch.randperm(len(t), generator=g).to(device)
            pred_sh = model.decode(ctx, q[perm])
        w = player_weight(pac_b, tgt.shape[-1], lw, device)
        tot["mse"] += F.mse_loss(pred, tgt).item()
        tot["weighted"] += weighted_mse(pred, tgt, w).item()
        tot["mse_shuffled"] += F.mse_loss(pred_sh, tgt).item()
        counts += torch.bincount(idx.cpu(), minlength=n)
        codes.append(idx.cpu())
    model.train()
    out = {k: v / len(batches) for k, v in tot.items()}
    out["code_gain"] = out["mse_shuffled"] / max(out["mse"], 1e-12)
    out["perplexity"] = perplexity(counts)
    out["usage"] = (counts.float() / counts.sum()).tolist()
    return out, torch.cat(codes)


@torch.no_grad()
def swap_grid(model, val, cfg, codec, device, out_path):
    """Rows: val contexts. Columns: last context frame | real target | decoded with code 0 .. n-1."""
    e, n = cfg["eval"], cfg["lam"]["n_codes"]
    model.eval()
    t = val.fixed_batches(e["swap_grid_contexts"], 1, seed=123)[0]
    ctx_u8, _, tgt_u8 = val.get_raw(t)
    ctx, tgt = to_dev(codec, ctx_u8, device), to_dev(codec, tgt_u8, device)
    cols = [ctx[:, -3:], tgt] + [model.decode(ctx, model.vq.embed[k].expand(len(t), -1)) for k in range(n)]
    model.train()
    s, size = e["upscale"], cfg["data"]["size"] * e["upscale"]
    names = ["last ctx", "real"] + [f"code {k}" for k in range(n)]
    sheet = Image.new("RGB", (len(cols) * (size + 4), 18 + len(t) * (size + 4)), "black")
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default(size=12)
    for c, (name, batch) in enumerate(zip(names, cols)):
        draw.text((c * (size + 4) + 2, 2), name, fill="white", font=font)
        for r in range(len(t)):
            sheet.paste(upscale(to_uint8(batch[r]).cpu(), s), (c * (size + 4), 18 + r * (size + 4)))
    sheet.save(out_path)


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]
    tr = dict(cfg["train"] if args.part == "lam" else {**cfg["train"], **cfg["prior"]})
    for k, v in (("steps", args.steps), ("batch_size", args.batch_size), ("eval_every", args.eval_every)):
        if v is not None:
            tr[k] = v
    if args.part == "lam" and d.get("action_source") != "none":
        raise SystemExit("the LAM trains on frames only: data.action_source must be 'none'")
    if args.part == "prior":
        d["action_source"] = "lam"                        # the prior's targets are the LAM's codes, never actions
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(tr["device"])
    use_bf16 = tr["bf16"] and device.type == "cuda"

    train, val, cache, (train_idx, val_idx) = get_datasets(cfg, args.firewall_scramble_seed)
    codec = FrameCodec(cache.get("palette")).to(device)
    pac = None
    if args.part == "lam" and cfg["lam"]["loss_weight"]["enabled"]:
        pac = torch.from_numpy(np.load(d["pac_positions"]))
        if len(pac) != int(cache["ep_start"][-1]):
            raise SystemExit(f"{d['pac_positions']} does not match the cache")
    print(f"episodes {len(train_idx)} train / {len(val_idx)} val; windows {len(train)} train / {len(val)} val; "
          f"part {args.part}; action slots: {d['action_source']}" + ("; player-weighted loss" if pac is not None else ""))

    model = (LAM(cfg) if args.part == "lam" else Prior(cfg)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.part} params: {n_params / 1e6:.2f}M on {device} (bf16={use_bf16})")
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    ckpt_dir = Path(tr["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"{args.part}.pt"
    step = 0
    if args.resume:
        ck = torch.load(ckpt, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"resumed from {ckpt} at step {step}")

    import wandb
    run = wandb.init(project=tr["wandb_project"], name=args.run_name, mode=args.wandb_mode,
                     config={**cfg, "part": args.part, "seed": args.seed, "params": n_params, "train_effective": tr})
    out_dir = Path(cfg["eval"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    val_batches = val.fixed_batches(tr["batch_size"], tr["eval_batches"], seed=args.seed)
    gen = torch.Generator().manual_seed(10_007 * args.seed + step + 1)
    batches = Batches(train, cfg, args.part, pac, tr["batch_size"], gen, tr.get("prefetch", False) and device.type == "cuda")
    dump = {"losses": []}

    def save():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": cfg}, ckpt)

    def evaluate():
        nonlocal val_codes
        if args.part == "lam":
            ev, val_codes = eval_lam(model, val, cfg, codec, pac, device, val_batches, use_bf16)
            grid = out_dir / f"swap_{step:06d}.png"
            swap_grid(model, val, cfg, codec, device, grid)
            logs = {f"val/{k}": v for k, v in ev.items() if k != "usage"}
            logs.update({f"val/usage_code{k}": u for k, u in enumerate(ev["usage"])})
            logs["val/swap_grid"] = wandb.Image(str(grid))
            print(f"  [eval @ {step}] mse {ev['mse']:.5f} | shuffled {ev['mse_shuffled']:.5f} (gain x{ev['code_gain']:.3f}) | "
                  f"perplexity {ev['perplexity']:.2f} | usage {[round(u, 3) for u in ev['usage']]} | restarts {model.vq.restarts}")
        else:
            model.eval()
            ce, acc, n, preds = 0.0, 0.0, 0, []
            with torch.no_grad():
                for t in val_batches:
                    ctx_u8, y = batches.get(t)
                    logits = model(to_dev(codec, ctx_u8, device))
                    y = y.to(device)
                    ce += F.cross_entropy(logits, y, reduction="sum").item()
                    acc += (logits.argmax(1) == y).sum().item()
                    n += len(y)
                    preds.append(logits.argmax(1).cpu())
            model.train()
            val_codes = torch.cat(preds)                  # the prior's predicted codes (compared by the firewall test)
            logs = {"val/ce": ce / n, "val/acc": acc / n}
            print(f"  [eval @ {step}] val ce {ce / n:.4f} | top-1 {acc / n:.3f}")
        wandb.log(logs, step=step)

    val_codes = None
    t_start = t_log = time.time()
    start_step = step
    model.train()
    end = min(tr["steps"], args.stop_at or tr["steps"])
    while step < end:
        lr = cosine_lr(step, tr)
        for pg in opt.param_groups:
            pg["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            if args.part == "lam":
                ctx_u8, enc_u8, tgt_u8, pac_b = batches.next()
                ctx, enc, tgt = to_dev(codec, ctx_u8, device), to_dev(codec, enc_u8, device), to_dev(codec, tgt_u8, device)
                pred, idx, commit, _ = model(enc, ctx)
                w = player_weight(pac_b, tgt.shape[-1], cfg["lam"]["loss_weight"], device)
                recon = weighted_mse(pred, tgt, w)
                tied = model.vq.calibrate_entropy(recon.item())      # lam.entropy.weight "tied": set on the first batch
                if tied is not None:
                    commit = commit + tied
                    print(f"entropy weight tied to the loss scale on the first batch: recon {recon.item():.6f} / "
                          f"|term| {abs(model.vq.entropy_raw.item()):.6f} = {float(model.vq.entropy_weight):.6f}")
                loss = recon + commit
                parts = {"recon": recon.item(), "commit": commit.item(), "mse": F.mse_loss(pred.detach(), tgt).item()}
                if model.vq.entropy:
                    parts["entropy_term"] = model.vq.entropy_weight.item() * model.vq.entropy_raw.item()
            else:
                ctx_u8, y = batches.next()
                loss = F.cross_entropy(model(to_dev(codec, ctx_u8, device)), y.to(device))
                parts = {}
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
        opt.step()
        step += 1
        if args.dump:
            dump["losses"].append([loss.item()] + list(parts.values()))
        if step % tr["log_every"] == 0:
            now = time.time()
            sps = tr["log_every"] / (now - t_log)
            t_log = now
            logs = {"train/loss": loss.item(), "train/lr": lr, "train/grad_norm": gnorm.item(), "train/steps_per_s": sps,
                    **{f"train/{k}": v for k, v in parts.items()}}
            if args.part == "lam":
                logs["train/batch_perplexity"] = perplexity(torch.bincount(idx.cpu(), minlength=cfg["lam"]["n_codes"]))
            wandb.log(logs, step=step)
            print(f"step {step:6d}/{tr['steps']}  loss {loss.item():.5f}  " + " ".join(f"{k} {v:.5f}" for k, v in parts.items())
                  + f"  gnorm {gnorm.item():.2f}  {sps:.1f} it/s  eta {(tr['steps'] - step) / max(sps, 1e-6) / 60:.0f} min")
        if step % tr["eval_every"] == 0 or step == end:
            evaluate()
            save()

    print(f"done: {step - start_step} steps in {(time.time() - t_start) / 60:.1f} min; checkpoint {ckpt}")
    if args.dump:
        dump["weights_sha256"] = weights_sha256(model)
        dump["val_codes"] = val_codes
        torch.save(dump, args.dump)
        print(f"wrote {args.dump}")
    run.finish()


if __name__ == "__main__":
    main()
