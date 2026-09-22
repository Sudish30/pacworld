"""Train Model 1 (EDM diffusion world model) with EMA, periodic evals, and checkpoints.

Every train.eval_every steps (using EMA weights):
  val denoising loss on fixed val batches with fixed sigma/noise draws
  outputs/model1/val_grid_{step}.png     last ctx | sample (grid_steps Euler) | real | abs error
  outputs/model1/rollout_{step}.gif      75-step autoregressive rollout (rollout_steps Euler) beside ground truth
  per-step rollout MSE logged as a wandb line series
Checkpoints: {checkpoint_dir}/model1_latest.pt (model, ema, opt, step) and model1_ema.pt (EMA weights only).
"""
import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from common import load_config
from dataset import FrameCodec, History, context_offsets, get_datasets, to_float, to_uint8
from model1 import build_model, count_params, euler_sample, sample_sigmas
from train_model0 import pick_device, psnr_from_mse, upscale


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/model1.yaml")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, help="override train.steps (smoke tests)")
    p.add_argument("--eval-every", type=int, help="override train.eval_every")
    p.add_argument("--resume", help="path to model1_latest.pt to continue from")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--run-name")
    return p.parse_args()


def noise_context(ctx, ccfg, generator, device, train=True):
    """GameNGen-style context noise. Returns (noised ctx, per-sample sigma)."""
    B = ctx.shape[0]
    if not ccfg["enabled"]:
        return ctx, torch.zeros(B, device=device)
    if train:
        use = (torch.rand(B, generator=generator) < ccfg["prob"]).float()
        log_s = torch.empty(B).uniform_(math.log(ccfg["sigma_min"]), math.log(ccfg["sigma_max"]), generator=generator)
        sigma = (log_s.exp() * use).to(device)
    else:
        sigma = torch.full((B,), float(ccfg["infer_sigma"]), device=device)
    noise = torch.randn(ctx.shape, generator=generator).to(device)
    return ctx + sigma[:, None, None, None] * noise, sigma


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.module.parameters(), model.parameters()):
            e.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)


@torch.no_grad()
def val_loss(model, val, batches, cfg, device, seed):
    """Denoising loss on fixed val batches with fixed sigma, noise, and context-noise draws."""
    d, ccfg = cfg["diffusion"], cfg["ctx_noise"]
    g = torch.Generator().manual_seed(seed)
    total = 0.0
    for idx in batches:
        ctx, acts, tgt = val.get(idx)
        ctx, acts, tgt = ctx.to(device), acts.to(device), tgt.to(device)
        ctx, ctx_sigma = noise_context(ctx, ccfg, g, device, train=True)
        sigma = sample_sigmas(d["p_mean"], d["p_std"], len(idx), "cpu", g).to(device)
        noise = torch.randn(tgt.shape, generator=g).to(device)
        total += model.loss(tgt, ctx, acts, sigma, ctx_sigma, noise).item()
    return total / len(batches)


@torch.no_grad()
def render_grid(model, val, cfg, device, out_path, seed):
    e, d, ccfg = cfg["eval"], cfg["diffusion"], cfg["ctx_noise"]
    n, s = e["grid_samples"], e["upscale"]
    idx = val.fixed_batches(n, 1, seed=123)[0]
    ctx, acts, tgt = val.get(idx)
    ctx, acts = ctx.to(device), acts.to(device)
    g = torch.Generator().manual_seed(seed)
    ctx_in, ctx_sigma = noise_context(ctx, ccfg, g, device, train=False)
    gd = torch.Generator(device=device).manual_seed(seed)
    pred = euler_sample(model, ctx_in, acts, ctx_sigma, d["grid_steps"], d, gd).cpu()
    mse = F.mse_loss(pred, tgt).item()
    rows = [("last ctx", ctx.cpu()[:, -3:]), ("sample", pred), ("real", tgt), ("|err|", (pred - tgt).abs().clamp(0, 2) - 1)]
    size, label_w = cfg["data"]["size"] * s, 70
    sheet = Image.new("RGB", (label_w + n * (size + 4), len(rows) * (size + 4)), "black")
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default(size=14)
    for r, (name, batch) in enumerate(rows):
        draw.text((4, r * (size + 4) + size // 2 - 8), name, fill="white", font=font)
        for c in range(n):
            sheet.paste(upscale(to_uint8(batch[c]), s), (label_w + c * (size + 4), r * (size + 4)))
    sheet.save(out_path)
    return mse


@torch.no_grad()
def render_rollout(model, cache, episode_idx, cfg, device, out_path, seed):
    e, d, ccfg = cfg["eval"], cfg["diffusion"], cfg["ctx_noise"]
    steps = int(e["rollout_seconds"] * e["fps"])
    start_abs, end_abs = int(cache["ep_start"][episode_idx]), int(cache["ep_start"][episode_idx + 1])
    t0 = min(start_abs + e["rollout_start"], end_abs - steps - 1)
    codec = FrameCodec(cache.get("palette"))
    frames = torch.from_numpy(codec.decode_np(cache["frames"][t0:t0 + steps]))   # the real frames being predicted, RGB
    actions = torch.from_numpy(cache["actions"][t0 - 1:t0 + steps - 1])  # actions[t] produced frames[t]
    names = ["NOOP", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT"]

    g = torch.Generator().manual_seed(seed)
    gd = torch.Generator(device=device).manual_seed(seed)
    episode = (codec.decode_np(cache["frames"][start_abs:end_abs]), cache["actions"][start_abs:end_abs])
    hist = History.from_episodes([episode], t0 - start_abs, context_offsets(cfg["data"]), device)
    preds, mses = [], []
    for t in range(steps):
        ctx, acts = hist.context(actions[t:t + 1].to(device))
        ctx_in, ctx_sigma = noise_context(ctx, ccfg, g, device, train=False)
        pred = euler_sample(model, ctx_in, acts, ctx_sigma, d["rollout_steps"], d, gd)
        real = to_float(frames[t].permute(2, 0, 1))[None].to(device)
        mses.append(F.mse_loss(pred, real).item())
        preds.append(to_uint8(pred[0]).cpu())
        hist.push(pred)

    s, size = e["upscale"], cfg["data"]["size"] * e["upscale"]
    font = ImageFont.load_default(size=12)
    gif = []
    for t in range(steps):
        canvas = Image.new("RGB", (2 * size + 8, size + 18), "black")
        canvas.paste(upscale(preds[t], s), (0, 18))
        canvas.paste(upscale(frames[t].permute(2, 0, 1), s), (size + 8, 18))
        dr = ImageDraw.Draw(canvas)
        dr.text((2, 2), f"pred  t={t + 1:<3d} {names[int(actions[t])]}", fill="white", font=font)
        dr.text((size + 10, 2), "real", fill="white", font=font)
        gif.append(canvas)
    gif[0].save(out_path, save_all=True, append_images=gif[1:], duration=int(1000 / e["fps"]), loop=0)
    return mses


def run_eval(step, model, ema, val, val_batches, cache, val_idx, cfg, device, seed, out_dir, wandb):
    import time as _t
    t = _t.time()
    net = ema.module
    vloss = val_loss(net, val, val_batches, cfg, device, seed)
    grid_path = out_dir / f"val_grid_{step:06d}.png"
    grid_mse = render_grid(net, val, cfg, device, grid_path, seed)
    gif_path = out_dir / f"rollout_{step:06d}.gif"
    mses = render_rollout(net, cache, val_idx[0], cfg, device, gif_path, seed)
    logs = {"val/denoise_loss": vloss, "val/grid_mse": grid_mse, "val/grid_psnr": psnr_from_mse(grid_mse),
            "rollout/mse_final": mses[-1], "rollout/mse_mean": float(np.mean(mses)),
            "rollout/mse_first": mses[0], "eval/val_grid": wandb.Image(str(grid_path)),
            "rollout/mse_curve": wandb.plot.line_series(xs=list(range(1, len(mses) + 1)), ys=[mses], keys=["mse"],
                                                        title=f"rollout mse (step {step})", xname="rollout step")}
    try:
        logs["rollout/video"] = wandb.Video(str(gif_path), format="gif")
    except Exception as ex:
        print(f"wandb video log skipped: {ex}")
    wandb.log(logs, step=step)
    print(f"  [eval @ {step}] val denoise {vloss:.4f} | grid mse {grid_mse:.4f} psnr {psnr_from_mse(grid_mse):.2f} dB | "
          f"rollout mse first/mean/final {mses[0]:.4f}/{np.mean(mses):.4f}/{mses[-1]:.4f} | {_t.time() - t:.0f}s")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    tr, d, ccfg = cfg["train"], cfg["diffusion"], cfg["ctx_noise"]
    if args.steps is not None:
        tr["steps"] = args.steps
    if args.eval_every is not None:
        tr["eval_every"] = args.eval_every
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(tr["device"])
    use_bf16 = tr["bf16"] and device.type == "cuda"

    train, val, cache, (train_idx, val_idx) = get_datasets(cfg)
    codec = train.codec.to(device)
    print(f"episodes {len(train_idx)} train / {len(val_idx)} val; windows {len(train)} train / {len(val)} val"
          + (f"; palette cache, {len(codec.palette)} colours expanded on {device}" if codec.palette is not None else ""))
    val_batches = val.fixed_batches(tr["batch_size"], tr["eval_batches"], seed=args.seed)

    model = build_model(cfg).to(device)
    n_params = count_params(model)
    print(f"model params: {n_params / 1e6:.2f}M on {device} (bf16={use_bf16})")
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    ema = EMA(model, tr["ema_decay"])
    step = 0
    init_from = (cfg.get("finetune") or {}).get("init_from")
    if init_from and not args.resume:
        # fine-tuning: weights (live and EMA) from another run, fresh optimizer, step 0. The source is only read.
        if Path(init_from).resolve().parent == Path(tr["checkpoint_dir"]).resolve():
            raise SystemExit("finetune.init_from must not live in this run's checkpoint_dir")
        ck = torch.load(init_from, map_location=device)
        if ck["cfg"]["data"].get("context_offsets") != cfg["data"].get("context_offsets") or ck["cfg"]["data"]["context"] != cfg["data"]["context"]:
            raise SystemExit("finetune.init_from was trained with a different context layout")
        model.load_state_dict(ck["model"])
        ema.module.load_state_dict(ck["ema"])
        print(f"fine-tuning from {init_from} (its step {ck['step']}); optimizer and step counter start fresh")
    ecfg = cfg.get("events")
    if ecfg:
        n_ev = train.set_event_targets(np.load(ecfg["index"]), ecfg["kinds"], ecfg["radius"])
        print(f"event windows: {dict(zip(ecfg['kinds'], n_ev))} targets within +-{ecfg['radius']} steps; {ecfg['frac']:.0%} of every batch, split evenly over the kinds")
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        ema.module.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"resumed from {args.resume} at step {step}")

    import wandb
    run = wandb.init(project=tr["wandb_project"], name=args.run_name, mode=args.wandb_mode,
                     config={**cfg, "seed": args.seed, "params": n_params, "resumed_from": args.resume})

    out_dir = Path(cfg["eval"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(tr["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save():
        torch.save({"model": model.state_dict(), "ema": ema.module.state_dict(), "opt": opt.state_dict(),
                    "step": step, "cfg": cfg}, ckpt_dir / "model1_latest.pt")
        torch.save({"ema": ema.module.state_dict(), "step": step, "cfg": cfg}, ckpt_dir / "model1_ema.pt")

    gen = torch.Generator().manual_seed(args.seed + step)
    t_start, t_log, start_step = time.time(), time.time(), step
    model.train()
    while step < tr["steps"]:
        lr = tr["lr"] * min(1.0, (step + 1) / tr["warmup_steps"])
        for pg in opt.param_groups:
            pg["lr"] = lr

        ctx_u8, acts, tgt_u8 = train.sample_mixed(tr["batch_size"], gen, ecfg["frac"]) if ecfg else train.sample(tr["batch_size"], gen)
        # the cached bytes cross the bus as they are stored (one byte per pixel for a palette cache) and the
        # RGB expansion happens on the GPU
        ctx = codec.decode(ctx_u8.to(device, non_blocking=True))
        tgt = codec.decode(tgt_u8.to(device, non_blocking=True)[:, None])
        acts = acts.to(device)
        ctx, ctx_sigma = noise_context(ctx, ccfg, gen, device, train=True)
        sigma = sample_sigmas(d["p_mean"], d["p_std"], tgt.shape[0], "cpu", gen).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            loss = model.loss(tgt, ctx, acts, sigma, ctx_sigma)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
        opt.step()
        ema.update(model)
        step += 1

        if step % tr["log_every"] == 0:
            now = time.time()
            sps = tr["log_every"] / (now - t_log)
            t_log = now
            eta_min = (tr["steps"] - step) / max(sps, 1e-6) / 60
            wandb.log({"train/loss": loss.item(), "train/lr": lr, "train/grad_norm": gnorm.item(),
                       "train/steps_per_s": sps, "elapsed_min": (now - t_start) / 60}, step=step)
            print(f"step {step:6d}/{tr['steps']}  loss {loss.item():.4f}  gnorm {gnorm.item():.2f}  "
                  f"{sps:.1f} it/s  eta {eta_min:.0f} min")

        if step % tr["eval_every"] == 0 or step == tr["steps"]:
            model.eval()
            run_eval(step, model, ema, val, val_batches, cache, val_idx, cfg, device, args.seed, out_dir, wandb)
            save()
            model.train()

    print(f"done: {step - start_step} steps in {(time.time() - t_start) / 60:.1f} min; checkpoints in {ckpt_dir}")
    run.finish()


if __name__ == "__main__":
    main()
