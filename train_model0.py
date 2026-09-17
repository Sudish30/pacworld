"""Train Model 0 (conv UNet, MSE) for a wall-clock budget, then render eval artifacts.

Artifacts (also logged to wandb):
  outputs/model0/val_grid.png    last context | prediction | ground truth | abs error, for val samples
  outputs/model0/rollout.gif     autoregressive rollout (prediction | real) on a val episode
"""
import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from common import load_config
from dataset import get_datasets, to_float, to_uint8
from model0 import build_model, count_params


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/model0.yaml")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--minutes", type=float, help="override train.minutes")
    p.add_argument("--steps", type=int, help="stop after this many steps instead of the time budget (smoke tests)")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--run-name")
    return p.parse_args()


def pick_device(name):
    if name == "mps" and not torch.backends.mps.is_available():
        print("warning: MPS not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(name)


def psnr_from_mse(mse):
    # tensors live in [-1, 1]; PSNR is conventionally quoted on [0, 1]
    return -10.0 * math.log10(max(mse / 4.0, 1e-12))


@torch.no_grad()
def evaluate(model, val, batches, device):
    model.eval()
    total, n = 0.0, 0
    for idx in batches:
        ctx, acts, tgt = val.get(idx)
        pred = model(ctx.to(device), acts.to(device))
        total += F.mse_loss(pred, tgt.to(device), reduction="sum").item()
        n += tgt.numel()
    model.train()
    mse = total / n
    return mse, psnr_from_mse(mse)


def upscale(u8_chw, s):
    img = Image.fromarray(u8_chw.permute(1, 2, 0).cpu().numpy())
    return img.resize((img.width * s, img.height * s), Image.NEAREST)


@torch.no_grad()
def render_grid(model, val, cfg, device, out_path):
    e = cfg["eval"]
    n, s = e["grid_samples"], e["upscale"]
    idx = val.fixed_batches(n, 1, seed=123)[0]
    ctx, acts, tgt = val.get(idx)
    pred = model(ctx.to(device), acts.to(device)).cpu()
    last = ctx[:, -3:]
    err = (pred - tgt).abs().clamp(0, 2) - 1.0  # abs error in [0,2] mapped onto the [-1,1] display range
    rows = [("last ctx", last), ("pred", pred), ("real", tgt), ("|err|", err)]
    size = cfg["data"]["size"] * s
    label_w = 70
    sheet = Image.new("RGB", (label_w + n * (size + 4), len(rows) * (size + 4)), "black")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=14)
    for r, (name, batch) in enumerate(rows):
        draw.text((4, r * (size + 4) + size // 2 - 8), name, fill="white", font=font)
        for c in range(n):
            sheet.paste(upscale(to_uint8(batch[c]), s), (label_w + c * (size + 4), r * (size + 4)))
    sheet.save(out_path)
    return sheet


@torch.no_grad()
def render_rollout(model, cache, val_episode_idx, cfg, device, out_path):
    e, K = cfg["eval"], cfg["data"]["context"]
    steps = int(e["rollout_seconds"] * e["fps"])
    start_abs = int(cache["ep_start"][val_episode_idx])
    end_abs = int(cache["ep_start"][val_episode_idx + 1])
    t0 = min(start_abs + e["rollout_start"], end_abs - steps - 1)
    frames = torch.from_numpy(cache["frames"][t0 - K:t0 + steps])           # (K+steps, H, W, 3) uint8
    actions = torch.from_numpy(cache["actions"][t0 - K:t0 + steps])
    names = ["NOOP", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT"]

    ctx = to_float(frames[:K].permute(0, 3, 1, 2)).reshape(1, K * 3, *frames.shape[1:3]).to(device)
    preds, mses = [], []
    for t in range(steps):
        acts = actions[t:t + K][None].to(device)
        pred = model(ctx, acts).clamp(-1, 1)
        real = to_float(frames[K + t].permute(2, 0, 1))[None].to(device)
        mses.append(F.mse_loss(pred, real).item())
        preds.append(to_uint8(pred[0]).cpu())
        ctx = torch.cat([ctx[:, 3:], pred], dim=1)

    s, size = e["upscale"], cfg["data"]["size"] * e["upscale"]
    font = ImageFont.load_default(size=12)
    gif_frames = []
    for t in range(steps):
        canvas = Image.new("RGB", (2 * size + 8, size + 18), "black")
        canvas.paste(upscale(preds[t], s), (0, 18))
        canvas.paste(upscale(frames[K + t].permute(2, 0, 1), s), (size + 8, 18))
        d = ImageDraw.Draw(canvas)
        d.text((2, 2), f"pred  t={t + 1:<3d} {names[int(actions[K + t - 1])]}", fill="white", font=font)
        d.text((size + 10, 2), "real", fill="white", font=font)
        gif_frames.append(canvas)
    gif_frames[0].save(out_path, save_all=True, append_images=gif_frames[1:], duration=int(1000 / e["fps"]), loop=0)
    return mses


def main():
    args = parse_args()
    cfg = load_config(args.config)
    tr = cfg["train"]
    if args.minutes is not None:
        tr["minutes"] = args.minutes
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(tr["device"])

    train, val, cache, (train_idx, val_idx) = get_datasets(cfg)
    print(f"episodes {len(train_idx)} train / {len(val_idx)} val; windows {len(train)} train / {len(val)} val")
    val_batches = val.fixed_batches(tr["batch_size"], tr["eval_batches"], seed=args.seed)

    model = build_model(cfg).to(device)
    n_params = count_params(model)
    print(f"model params: {n_params / 1e6:.2f}M on {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])

    import wandb
    run = wandb.init(project=tr["wandb_project"], name=args.run_name, mode=args.wandb_mode,
                     config={**cfg, "seed": args.seed, "params": n_params, "steps_override": args.steps})

    budget_s = tr["minutes"] * 60
    gen = torch.Generator().manual_seed(args.seed)
    out_dir = Path(cfg["eval"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(tr["checkpoint"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    def progress(step, elapsed):
        return step / args.steps if args.steps else elapsed / budget_s

    step, best_val, t_start, t_log = 0, float("inf"), time.time(), time.time()
    model.train()
    while True:
        elapsed = time.time() - t_start
        frac = progress(step, elapsed)
        if frac >= 1.0:
            break
        lr = tr["lr"] * 0.5 * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = lr

        ctx, acts, tgt = train.sample(tr["batch_size"], gen)
        pred = model(ctx.to(device), acts.to(device))
        loss = F.mse_loss(pred, tgt.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
        opt.step()
        step += 1

        if step % tr["log_every"] == 0:
            now = time.time()
            sps = tr["log_every"] / (now - t_log)
            t_log = now
            wandb.log({"train/loss": loss.item(), "train/psnr": psnr_from_mse(loss.item()), "train/lr": lr,
                       "train/steps_per_s": sps, "elapsed_min": elapsed / 60}, step=step)
            print(f"step {step:6d}  loss {loss.item():.5f}  lr {lr:.2e}  {sps:.1f} it/s  {elapsed / 60:.1f} min")

        if step % tr["eval_every"] == 0:
            vmse, vpsnr = evaluate(model, val, val_batches, device)
            wandb.log({"val/mse": vmse, "val/psnr": vpsnr}, step=step)
            print(f"  val mse {vmse:.5f}  psnr {vpsnr:.2f} dB")
            if vmse < best_val:
                best_val = vmse
                torch.save({"model": model.state_dict(), "cfg": cfg, "step": step, "val_mse": vmse}, ckpt_path)

    vmse, vpsnr = evaluate(model, val, val_batches, device)
    print(f"final: {step} steps in {(time.time() - t_start) / 60:.1f} min, val mse {vmse:.5f} psnr {vpsnr:.2f} dB")
    wandb.log({"val/mse": vmse, "val/psnr": vpsnr}, step=step)
    if vmse <= best_val or not ckpt_path.exists():
        torch.save({"model": model.state_dict(), "cfg": cfg, "step": step, "val_mse": vmse}, ckpt_path)
    print(f"checkpoint: {ckpt_path}")

    model.eval()
    grid_path = out_dir / "val_grid.png"
    render_grid(model, val, cfg, device, grid_path)
    print(f"wrote {grid_path}")
    gif_path = out_dir / "rollout.gif"
    mses = render_rollout(model, cache, val_idx[0], cfg, device, gif_path)
    print(f"wrote {gif_path}: {len(mses)} steps, rollout mse first/mid/last = {mses[0]:.4f}/{mses[len(mses) // 2]:.4f}/{mses[-1]:.4f}")
    logs = {"eval/val_grid": wandb.Image(str(grid_path)), "eval/rollout_mse_final": mses[-1],
            "eval/rollout_mse_curve": wandb.plot.line_series(xs=list(range(1, len(mses) + 1)), ys=[mses],
                                                              keys=["mse"], title="rollout mse", xname="step")}
    try:
        logs["eval/rollout"] = wandb.Video(str(gif_path), format="gif")
    except Exception as ex:  # video logging is best-effort
        print(f"wandb video log skipped: {ex}")
    wandb.log(logs, step=step)
    run.finish()


if __name__ == "__main__":
    main()
