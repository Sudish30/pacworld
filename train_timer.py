"""Train one cell (layout, N) of the hidden-timer experiment (notes/timer_rule_design.md).

The pacworld diffusion recipe, small and unconditional: the same EDM denoiser, context-noise augmentation, EMA and
context rule (dataset.WindowDataset / gather_context), on in-memory episodes of timer_game.py. The only variable
between layouts is the context offsets. Checkpoint: <train.checkpoint_root>/<layout>-N<N>/model.pt (EMA weights).

  python train_timer.py --config configs/timer.yaml --layout S10 --N 33 --seed 0
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from common import load_config
from dataset import FrameCodec, WindowDataset
from model1 import build_model, count_params, sample_sigmas
from timer_game import generate_split
from train_model0 import pick_device
from train_model1 import EMA, noise_context


def as_cache(eps):
    frames = np.concatenate([fr for fr, _, _, _ in eps])
    starts = np.concatenate([[0], np.cumsum([len(fr) for fr, _, _, _ in eps])]).astype(np.int64)
    acts = np.zeros(len(frames), np.int64)
    acts[starts[1:] - 1] = -1
    return {"frames": frames, "actions": acts, "ep_start": starts}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/timer.yaml")
    p.add_argument("--layout", required=True)
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, help="override train.steps (pilots only)")
    p.add_argument("--batch-size", type=int, help="override train.batch_size (pilots only)")
    p.add_argument("--checkpoint", help="override the checkpoint path (pilots only)")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    a = p.parse_args()
    cfg = load_config(a.config)
    tr = cfg["train"]
    for k, v in (("steps", a.steps), ("batch_size", a.batch_size)):
        if v is not None:
            tr[k] = v
    cells = cfg["grid"].get(a.layout, []) + cfg.get("followup_grid", {}).get(a.layout, [])
    if a.N not in cells:
        raise SystemExit(f"N={a.N} is not a pre-registered cell of {a.layout}")
    offsets = cfg["layouts"][a.layout]
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = pick_device(tr["device"])
    use_bf16 = tr["bf16"] and device.type == "cuda"

    train_eps, train_sha = generate_split(cfg["game"], a.N, "train")
    val_eps, _ = generate_split(cfg["game"], a.N, "val")
    tcache, vcache = as_cache(train_eps), as_cache(val_eps)
    codec = FrameCodec(None)
    train = WindowDataset(tcache, range(len(train_eps)), offsets, codec)
    val = WindowDataset(vcache, range(len(val_eps)), offsets, codec)
    mcfg = {"model": cfg["model"], "data": {"context": len(offsets), "context_offsets": offsets, "size": cfg["game"]["size"]},
            "diffusion": cfg["diffusion"], "ctx_noise": cfg["ctx_noise"], "train": tr, "layout": a.layout, "N": a.N,
            "train_sha256": train_sha}
    model = build_model(mcfg).to(device)
    ema = EMA(model, tr["ema_decay"])
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    n_params = count_params(model)
    print(f"{a.layout} N={a.N}: {len(train)} train windows, {count_params(model) / 1e6:.2f}M params on {device}; train sha256 {train_sha[:16]}")

    import wandb
    run = wandb.init(project=tr["wandb_project"], group=tr["wandb_group"], name=f"timer-{a.layout}-N{a.N}", mode=a.wandb_mode,
                     config={**mcfg, "seed": a.seed, "params": n_params})
    ck = Path(a.checkpoint or Path(tr["checkpoint_root"]) / f"{a.layout}-N{a.N}" / "model.pt")
    ck.parent.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(a.seed)
    d, ccfg = cfg["diffusion"], cfg["ctx_noise"]
    val_batches = val.fixed_batches(tr["batch_size"], tr["val_batches"], seed=a.seed)

    @torch.no_grad()
    def val_loss():
        g = torch.Generator().manual_seed(a.seed)
        tot = 0.0
        for idx in val_batches:
            ctx, acts, tgt = val.get(idx)
            ctx, acts, tgt = ctx.to(device), acts.clamp(min=0).to(device), tgt.to(device)
            ctx, cs = noise_context(ctx, ccfg, g, device, train=True)
            sig = sample_sigmas(d["p_mean"], d["p_std"], len(idx), "cpu", g).to(device)
            tot += ema.module.loss(tgt, ctx, acts, sig, cs, torch.randn(tgt.shape, generator=g).to(device)).item()
        return tot / len(val_batches)

    t0 = t_log = time.time()
    for step in range(1, tr["steps"] + 1):
        lr = (tr["lr"] * step / tr["warmup_steps"] if step <= tr["warmup_steps"] else
              tr["lr_min"] + 0.5 * (tr["lr"] - tr["lr_min"]) * (1 + math.cos(math.pi * (step - tr["warmup_steps"]) / (tr["steps"] - tr["warmup_steps"]))))
        for pg in opt.param_groups:
            pg["lr"] = lr
        ctx_u8, acts, tgt_u8 = train.sample(tr["batch_size"], gen)
        ctx, tgt = codec.decode(ctx_u8.to(device)), codec.decode(tgt_u8.to(device)[:, None])
        acts = acts.clamp(min=0).to(device)
        ctx, cs = noise_context(ctx, ccfg, gen, device, train=True)
        sig = sample_sigmas(d["p_mean"], d["p_std"], len(tgt), "cpu", gen).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            loss = model.loss(tgt, ctx, acts, sig, cs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
        opt.step()
        ema.update(model)
        if step % tr["log_every"] == 0:
            sps = tr["log_every"] / (time.time() - t_log)
            t_log = time.time()
            wandb.log({"train/loss": loss.item(), "train/lr": lr, "train/grad_norm": gn.item(), "train/steps_per_s": sps}, step=step)
            print(f"step {step}/{tr['steps']} loss {loss.item():.4f} {sps:.1f} it/s")
        if step % tr["eval_every"] == 0 or step == tr["steps"]:
            vl = val_loss()
            wandb.log({"val/denoise_loss": vl}, step=step)
            print(f"  [eval @ {step}] val denoise {vl:.5f}")
            torch.save({"ema": ema.module.state_dict(), "step": step, "cfg": mcfg}, ck)
    print(f"done: {tr['steps']} steps in {(time.time() - t0) / 60:.1f} min -> {ck}")
    run.finish()


if __name__ == "__main__":
    main()
