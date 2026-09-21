"""128x128 speed test of the demo path and the training step (untrained weights)."""
import sys, time, copy, io, numpy as np, torch
from PIL import Image
sys.path.insert(0, ".")
from common import load_config
from model1 import build_model, count_params, euler_sample, sample_sigmas
from dataset import to_uint8
dev = torch.device("cuda")
base = load_config("configs/m1-2M-ctx6s16.yaml")
variants = [("64px current (reference)", 64, [64, 96, 192, 192], [2, 3]),
            ("128px A: same 4 levels, attention at 32x32 + 16x16", 128, [64, 96, 192, 192], [2, 3]),
            ("128px B: 5 levels [64,64,96,192,192], attention at 16x16 + 8x8", 128, [64, 64, 96, 192, 192], [3, 4])]
for name, size, widths, attn in variants:
    cfg = copy.deepcopy(base); cfg["model"]["widths"], cfg["model"]["attn_levels"], cfg["data"]["size"] = widths, attn, size
    K, d = cfg["data"]["context"], cfg["diffusion"]
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = build_model(cfg).to(dev).eval()
    gen = torch.Generator(device=dev).manual_seed(0)
    ctx = torch.randn(1, 3 * K, size, size, device=dev); acts = torch.randint(0, 9, (1, K), device=dev); sig = torch.full((1,), 0.01, device=dev)
    lat = []
    with torch.inference_mode():
        for i in range(130):
            t0 = time.perf_counter()
            c = ctx + 0.01 * torch.randn(ctx.shape, device=dev, generator=gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = euler_sample(model, c, acts, sig, 3, d, gen)
            frame = to_uint8(pred.float()[0]).permute(1, 2, 0).cpu().numpy()
            buf = io.BytesIO(); Image.fromarray(frame).save(buf, format="PNG", compress_level=1)
            if i >= 30: lat.append(time.perf_counter() - t0)
    lat = 1000 * np.array(lat); p50, p95 = np.percentile(lat, 50), np.percentile(lat, 95)
    infer_mem = torch.cuda.max_memory_allocated() / 2**30
    # training step, batch 64, bf16, AdamW + EMA
    model.train(); ema = copy.deepcopy(model); opt = torch.optim.AdamW(model.parameters(), lr=1e-4); B = cfg["train"]["batch_size"]
    torch.cuda.reset_peak_memory_stats()
    def step():
        tgt = torch.randn(B, 3, size, size, device=dev); cx = torch.randn(B, 3 * K, size, size, device=dev)
        a = torch.randint(0, 9, (B, K), device=dev); s = sample_sigmas(-0.4, 1.2, B, "cpu", None).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss(tgt, cx, a, s, torch.zeros(B, device=dev))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        for e, q in zip(ema.parameters(), model.parameters()): e.data.mul_(0.999).add_(q.detach(), alpha=0.001)
    try:
        for _ in range(5): step()
        torch.cuda.synchronize(); t = time.time()
        for _ in range(25): step()
        torch.cuda.synchronize(); its = 25 / (time.time() - t); train_mem = torch.cuda.max_memory_allocated() / 2**30
        tr = f"{its:.2f} it/s at batch {B}, peak VRAM {train_mem:.1f} GiB"
    except torch.OutOfMemoryError:
        its, tr = float("nan"), f"OOM at batch {B}"
    print(f"{name}\n   params {count_params(model) / 1e6:.2f}M | demo path (3 Euler steps, bf16, PNG encode): p50 {p50:.1f} ms, p95 {p95:.1f} ms -> max {1000 / p50:.1f} fps "
          f"({'fits' if p95 < 1000 / 15 else 'does NOT fit'} the 66.7 ms budget of 15 fps; inference VRAM {infer_mem:.1f} GiB) | training: {tr}")
    del model, ema, opt
