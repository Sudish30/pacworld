"""Export the served world model for the browser demo and check it against PyTorch.

Writes to web/ (configs/web_demo.yaml):
  model_fp32.onnx / model_fp16.onnx  one graph = the whole Euler sampler of model1.euler_sample:
        inputs  ctx (1, 3K, H, W) float32 in [-1, 1] (context noise already added), actions (1, K) int64,
                noise (1, 3, H, W) float32 ~ N(0, 1). The context-noise level is fixed in the graph (ctx_sigma).
        output  frame (1, 3, H, W) float32 in [-1, 1]
  cases.bin + cases.json   fixed test cases from held-out episodes with PyTorch's fp32 output for each
  starts.bin + starts.json real histories (uint8 frames and actions) the demo starts from
Then runs both ONNX files with ONNX Runtime on the CPU and prints the fidelity numbers of the pass bar.

  python tools/export_onnx.py --config configs/web_demo.yaml --seed 0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from common import load_config  # noqa: E402
from dataset import History, context_offsets, episode_files, to_uint8  # noqa: E402
from model1 import build_model, euler_sample, karras_schedule  # noqa: E402
import detectors as D  # noqa: E402


class Sampler(nn.Module):
    """model1.euler_sample for one fixed context-noise level, with the starting noise as an input.

    Everything that depends only on the noise levels is computed here once, in float32, and stored as constants: the
    EDM coefficients of each step and the two noise-level embeddings (Fourier features -> projection). The learned
    Fourier frequencies reach about 42, so the angles reach hundreds of radians, which 16-bit floats cannot resolve;
    the PyTorch server keeps sin/cos in float32 under bf16 autocast, and so does this. The Euler step is written as
    x + r * (x - D) with r = (s_next - s) / s, which avoids dividing by the smallest sigma at run time.
    """

    def __init__(self, denoiser, n_steps, dcfg, ctx_sigma):
        super().__init__()
        self.unet = denoiser.unet
        sig = karras_schedule(n_steps, dcfg["sigma_min"], dcfg["sigma_max"], dcfg["rho"], "cpu")
        self.sigma0 = float(sig[0])
        self.sigmas, self.ctx_sigma = [float(v) for v in sig], float(ctx_sigma)
        self.steps, self.pad = [], 0
        with torch.no_grad():
            cs = torch.full((1,), float(ctx_sigma))
            ctx_emb = self.unet.ctx_proj(self.unet.ctx_ff(denoiser.ctx_noise_embedding(cs)))
            for k in range(n_steps):
                s, s_next = sig[k:k + 1], float(sig[k + 1])
                c_skip, c_out, c_in, c_noise = denoiser.coefficients(s)
                self.register_buffer(f"emb{k}", self.unet.noise_proj(self.unet.noise_ff(c_noise)) + ctx_emb)
                self.steps.append((float(c_skip), float(c_out), float(c_in), (s_next - float(s)) / float(s)))

    def pad_in_conv(self, to):
        """Give the first convolution `to` input channels (extra ones are zero-weighted and fed zeros). ONNX Runtime
        Web 1.30.0's WebGPU backend computed the 33-channel input convolution wrongly (web/debug.html: first node to
        differ from its CPU backend, relative error 0.75); with a channel count divisible by 4 it is exact."""
        old = self.unet.in_conv
        self.pad = to - old.in_channels
        new = nn.Conv2d(to, old.out_channels, old.kernel_size, padding=old.padding)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :old.in_channels] = old.weight
            new.bias.copy_(old.bias)
        self.unet.in_conv = new

    def forward(self, ctx, actions, noise):
        if self.pad:
            ctx = torch.cat([ctx, torch.zeros_like(ctx[:, :self.pad])], dim=1)
        x = noise * self.sigma0
        act = self.unet.action_proj(self.unet.action_embed(actions).flatten(1))
        for k, (c_skip, c_out, c_in, r) in enumerate(self.steps):
            cond = self.unet.cond_mlp(act + getattr(self, f"emb{k}"))
            den = c_skip * x + c_out * self.unet.run(torch.cat([c_in * x, ctx], dim=1), cond)
            x = x + r * (x - den)
        return x.clamp(-1, 1)


def plain_attention(self, x):
    """model1.Attention.forward with the attention written out (softmax(q k^T / sqrt(d)) v). The exporter turns
    F.scaled_dot_product_attention into a graph with float32 casts that break the 16-bit conversion; this is the same
    computation, and the wrapper is checked against model1.euler_sample (which uses the original) before exporting."""
    B, C, H, W = x.shape
    q, k, v = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads, H * W).unbind(1)
    q, k, v = q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2)
    w = torch.softmax(q @ k.transpose(-1, -2) * (C // self.heads) ** -0.5, dim=-1)
    return x + self.proj((w @ v).transpose(-1, -2).reshape(B, C, H, W))


def euler_sample_fixed(model, sampler, ctx, act, noise):
    """model1's own denoiser stepped with the given starting noise (runs under whatever autocast is active)."""
    x = torch.from_numpy(noise) * sampler.sigma0
    ctx, act = torch.from_numpy(ctx), torch.from_numpy(act)
    cs = torch.full((1,), sampler.ctx_sigma)
    for s, s_next in zip(sampler.sigmas[:-1], sampler.sigmas[1:]):
        d = (x - model(x, torch.full((1,), s), ctx, act, cs).float()) / s
        x = x + (s_next - s) * d
    return x.clamp(-1, 1)


def prng_normal(seed, n):
    """n standard normals from mulberry32 + Box-Muller; web/pacworld.js has the same function (seededNormal)."""
    m = (n + 1) // 2 * 2
    with np.errstate(over="ignore"):
        a = (np.uint32(seed) + np.uint32(0x6D2B79F5) * np.arange(1, m + 1, dtype=np.uint32)).astype(np.uint32)
        t = (a ^ (a >> np.uint32(15))) * (a | np.uint32(1))
        t = t.astype(np.uint32)
        t = (t ^ (t + ((t ^ (t >> np.uint32(7))) * (t | np.uint32(61))).astype(np.uint32))).astype(np.uint32)
        u = ((t ^ (t >> np.uint32(14))).astype(np.uint32)).astype(np.float64) / 4294967296.0
    u1, u2 = 1.0 - u[0::2], u[1::2]
    r, th = np.sqrt(-2.0 * np.log(u1)), 2.0 * np.pi * u2
    return np.stack([r * np.cos(th), r * np.sin(th)], 1).reshape(-1)[:n].astype(np.float32)


def plain_groupnorm(self, x):
    """nn.GroupNorm written with basic operations. The exporter turns GroupNorm into InstanceNormalization, which the
    WebGPU backend of ONNX Runtime Web 1.30.0 computed wrongly for this model (frames off by ~10/255 while the same
    graph was exact on its CPU backend); mean / variance / multiply are exact there."""
    B, C, H, W = x.shape
    g = x.reshape(B, self.num_groups, -1)
    mean = g.mean(-1, keepdim=True)
    var = (g - mean).pow(2).mean(-1, keepdim=True)
    y = ((g - mean) * (var + self.eps).rsqrt()).reshape(B, C, H, W)
    return y * self.weight[None, :, None, None] + self.bias[None, :, None, None] if self.affine else y


def plain_upsample(self, x):
    """model1.Upsample with the nearest-neighbour x2 written as a reshape (no Resize operator)."""
    B, C, H, W = x.shape
    return self.conv(x[:, :, :, None, :, None].expand(B, C, H, 2, W, 2).reshape(B, C, 2 * H, 2 * W))


def longest_val_episodes(cfg, n):
    val = set(json.load(open(ROOT / cfg["data"]["val_episodes"]))["val_episode_seeds"])
    files = [f for f in episode_files(cfg["data"], ROOT) if int(f.stem.split("_")[1]) in val]
    eps = []
    for f in files:
        z = np.load(f)
        eps.append((len(z["actions"]), int(f.stem.split("_")[1]), f))
    eps.sort(key=lambda t: (-t[0], t[1]))
    out = []
    for _, seed, f in eps[:n]:
        z = np.load(f)
        out.append((seed, D.downsample_frames(z["frames"], cfg["data"]["size"], cfg["data"]["resample"]), z["actions"].astype(np.int64)))
    return out


def fidelity(a, b, fd, baseline=None):
    """Per-case comparison of uint8 frames (n, 3, S, S). A case passes if its mean |diff| and its share of values
    differing by more than fd.big_diff are within the bar; a case over the bar is excused only if PyTorch's own bf16
    autocast (what the server runs) misses the fp32 frame of that case by at least as much (`baseline`)."""
    d = np.abs(a.astype(np.int32) - b.astype(np.int32)).reshape(len(a), -1)
    mean, big = d.mean(1), (d > fd["big_diff"]).mean(1)
    ok = (mean <= fd["max_mean_abs_diff"]) & (big <= fd["max_big_diff_frac"])
    excused = np.zeros(len(a), bool)
    if baseline is not None:
        excused = ~ok & (np.array(baseline["per_case_mean"]) >= mean) & (np.array(baseline["per_case_big"]) >= big)
    return {"mean_abs_diff": float(d.mean()), "max_abs_diff": int(d.max()), "big_diff_frac": float((d > fd["big_diff"]).mean()),
            "worst_case_mean": float(mean.max()), "worst_case_big": float(big.max()), "cases_over_bar": int((~ok).sum()),
            "cases_excused_by_bf16": int(excused.sum()), "per_case_mean": [float(v) for v in mean],
            "per_case_big": [float(v) for v in big], "pass": bool((ok | excused).all())}


def main():  # noqa: C901
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/web_demo.yaml")
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    cfg = load_config(ROOT / a.config)
    torch.manual_seed(a.seed)
    out = ROOT / cfg["out_dir"]
    out.mkdir(exist_ok=True)
    ck = torch.load(ROOT / cfg["checkpoint"], map_location="cpu")
    model = build_model(ck["cfg"])
    model.load_state_dict(ck["ema"])
    model.eval()
    offsets = context_offsets(ck["cfg"]["data"])
    K, S = len(offsets), cfg["data"]["size"]
    import copy
    import types
    from model1 import Attention, Upsample
    export_model = copy.deepcopy(model)
    for mod in export_model.modules():
        if isinstance(mod, Attention):
            mod.forward = types.MethodType(plain_attention, mod)
        elif isinstance(mod, nn.GroupNorm) and "groupnorm" in cfg["plain_ops"]:
            mod.forward = types.MethodType(plain_groupnorm, mod)
        elif isinstance(mod, Upsample) and "upsample" in cfg["plain_ops"]:
            mod.forward = types.MethodType(plain_upsample, mod)
    sampler = Sampler(export_model, cfg["sampler_steps"], ck["cfg"]["diffusion"], cfg["ctx_sigma"]).eval()
    if cfg.get("in_conv_channels"):
        sampler.pad_in_conv(cfg["in_conv_channels"])
        sampler.eval()

    # ---- fixed test cases and PyTorch references
    c = cfg["cases"]
    eps = longest_val_episodes(cfg, max(c["episodes"], cfg["starts"]["episodes"]))
    gen = torch.Generator().manual_seed(a.seed)
    ctxs, acts, noises, refs, meta = [], [], [], [], []
    sig = torch.full((1,), float(cfg["ctx_sigma"]))
    k = 0
    while len(ctxs) < c["n"]:
        seed, frames, actions = eps[len(ctxs) % c["episodes"]]
        t = c["first_step"] + (len(ctxs) // c["episodes"]) * c["step_stride"]
        hist = History.from_episodes([(frames, actions)], t, offsets, "cpu")
        ctx, act = hist.context(torch.tensor([int(actions[t - 1])]))
        ctx = ctx + cfg["ctx_sigma"] * torch.randn(ctx.shape, generator=gen)      # the server's context noise
        noise = torch.randn(1, 3, S, S, generator=gen)
        with torch.no_grad():
            ref = sampler(ctx, act, noise)
            if k == 0:                                                           # the wrapper equals euler_sample
                g = torch.Generator().manual_seed(123)
                n0 = torch.randn(1, 3, S, S, generator=torch.Generator().manual_seed(123))
                same = torch.allclose(sampler(ctx, act, n0), euler_sample(model, ctx, act, sig, cfg["sampler_steps"], ck["cfg"]["diffusion"], g), atol=1e-4)
                print(f"wrapper == model1.euler_sample on the same noise: {same}")
                if not same:
                    raise SystemExit("the exported sampler differs from euler_sample")
        ctxs.append(ctx.numpy()); acts.append(act.numpy()); noises.append(noise.numpy()); refs.append(to_uint8(ref[0]).numpy())
        meta.append({"episode": seed, "target_step": t})
        k += 1
    ctxs, acts, noises, refs = np.concatenate(ctxs), np.concatenate(acts), np.concatenate(noises), np.stack(refs)

    # ---- export
    ex = (torch.from_numpy(ctxs[:1]), torch.from_numpy(acts[:1]), torch.from_numpy(noises[:1]))
    f32 = out / cfg["model_fp32"]
    torch.onnx.export(sampler, ex, str(f32), input_names=["ctx", "actions", "noise"], output_names=["frame"],
                      opset_version=cfg["opset"], dynamo=False, do_constant_folding=True)
    import onnx
    from onnxconverter_common import float16
    own_casts = {n.name for n in onnx.load(str(f32)).graph.node if n.op_type == "Cast"}
    m16 = float16.convert_float_to_float16(onnx.load(str(f32)), keep_io_types=True)
    for node in m16.graph.node:          # the model's own casts to float32 (from .float() calls) become float16;
        if node.op_type == "Cast" and node.name in own_casts:      # casts the converter added are left alone
            for at in node.attribute:
                if at.name == "to" and at.i == onnx.TensorProto.FLOAT:
                    at.i = onnx.TensorProto.FLOAT16
    f16 = out / cfg["model_fp16"]
    onnx.save(m16, str(f16))
    ops = sorted({n.op_type for n in onnx.load(str(f32)).graph.node})
    print(f"exported {f32.name} ({f32.stat().st_size / 1e6:.1f} MB) and {f16.name} ({f16.stat().st_size / 1e6:.1f} MB); ops: {ops}")

    # ---- baselines and checks in PyTorch
    fd, res = cfg["fidelity"], {}
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):      # what the server runs (bf16 autocast)
        bf = np.stack([to_uint8(euler_sample_fixed(model, sampler, ctxs[i:i + 1], acts[i:i + 1], noises[i:i + 1])[0]).numpy()
                       for i in range(len(ctxs))])
    res["pytorch_bf16"] = fidelity(bf, refs, fd)
    peak = {"v": 0.0}
    hooks = [m.register_forward_hook(lambda _m, _i, o: peak.__setitem__("v", max(peak["v"], float(o.abs().max()))))
             for m in sampler.modules() if isinstance(m, (nn.Conv2d, nn.Linear, nn.GroupNorm))]
    with torch.no_grad():
        for i in range(len(ctxs)):
            sampler(torch.from_numpy(ctxs[i:i + 1]), torch.from_numpy(acts[i:i + 1]), torch.from_numpy(noises[i:i + 1]))
    for h in hooks:
        h.remove()
    res["peak_activation_fp32"] = peak["v"]
    print(f"PyTorch bf16 autocast vs fp32: mean |diff| {res['pytorch_bf16']['mean_abs_diff']:.4f}, worst case "
          f"{res['pytorch_bf16']['worst_case_mean']:.4f}; largest activation in fp32 {peak['v']:.1f} (fp16 limit 65504)")

    # ---- ONNX Runtime (CPU) against PyTorch
    import onnxruntime as ort
    sessions = {}
    for name, path in (("fp32", f32), ("fp16", f16)):
        sess = sessions[name] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        outs = [sess.run(None, {"ctx": ctxs[i:i + 1], "actions": acts[i:i + 1], "noise": noises[i:i + 1]})[0][0] for i in range(len(ctxs))]
        if not all(np.isfinite(o).all() for o in outs):
            raise SystemExit(f"{name}: non-finite output")
        u8 = np.stack([to_uint8(torch.from_numpy(o)).numpy() for o in outs])
        r = res[name] = fidelity(u8, refs, fd, res["pytorch_bf16"])
        print(f"ONNX Runtime CPU {name} vs PyTorch fp32 on {len(ctxs)} cases: mean |diff| {r['mean_abs_diff']:.4f}, max "
              f"{r['max_abs_diff']}, worst case mean {r['worst_case_mean']:.4f}, worst case share > {fd['big_diff']}: "
              f"{r['worst_case_big']:.5f}, cases over the bar {r['cases_over_bar']} -> {'pass' if r['pass'] else 'FAIL'}")

    # ---- same-noise rollout (diagnostic): PyTorch fp32 frames for the page to compare against, noise from a PRNG
    # that web/pacworld.js implements identically (mulberry32 + Box-Muller)
    ro = cfg["rollout_check"]
    seed_ep, frames, actions = eps[0]
    hist = History.from_episodes([(frames, actions)], cfg["starts"]["start_step"], offsets, "cpu")
    ro_frames, ro_actions = [], []
    hist16 = History.from_episodes([(frames, actions)], cfg["starts"]["start_step"], offsets, "cpu")
    first_bad = None
    for j in range(ro["steps"]):
        action = int(ro["actions"][(j // ro["hold"]) % len(ro["actions"])])
        ctx, act = hist.context(torch.tensor([action]))
        cn = torch.from_numpy(prng_normal(ro["seed"] * 100003 + 2 * j, ctx.numel())).reshape(ctx.shape)
        nz = torch.from_numpy(prng_normal(ro["seed"] * 100003 + 2 * j + 1, 3 * S * S)).reshape(1, 3, S, S)
        with torch.no_grad():
            pred = sampler(ctx + cfg["ctx_sigma"] * cn, act, nz)
        hist.push(pred)
        ro_frames.append(to_uint8(pred[0]).numpy()); ro_actions.append(action)
        c16, a16 = hist16.context(torch.tensor([action]))
        p16 = torch.from_numpy(sessions["fp16"].run(None, {"ctx": (c16 + cfg["ctx_sigma"] * cn).numpy(), "actions": a16.numpy(), "noise": nz.numpy()})[0])
        hist16.push(p16)
        d = np.abs(to_uint8(p16[0]).numpy().astype(int) - ro_frames[-1].astype(int))
        if first_bad is None and (d.mean() > fd["max_mean_abs_diff"] or (d > fd["big_diff"]).mean() > fd["max_big_diff_frac"]):
            first_bad = j + 1
    res["rollout_fp16_cpu_first_step_over_bar"] = first_bad
    print(f"same-noise {ro['steps']}-step rollout, ONNX fp16 (CPU) vs PyTorch fp32: first step over the bar: {first_bad}")
    with open(out / ro["file"], "wb") as fh:
        fh.write(np.stack(ro_frames).astype(np.uint8).tobytes())

    # ---- files for the page
    with open(out / c["file"], "wb") as fh:
        for arr in (ctxs.astype(np.float32), acts.astype(np.int32), noises.astype(np.float32), refs.astype(np.uint8)):
            fh.write(arr.tobytes())
    json.dump({"n": len(ctxs), "K": K, "size": S, "ctx_sigma": cfg["ctx_sigma"], "order": ["ctx f32 (n,3K,S,S)", "actions i32 (n,K)",
               "noise f32 (n,3,S,S)", "ref u8 (n,3,S,S)"], "cases": meta, "fidelity_bar": fd, "python_check": res, "rollout": {**cfg["rollout_check"], "episode": seed_ep}},
              open(out / "cases.json", "w"), indent=1)
    st, reach = cfg["starts"], -min(offsets)
    with open(out / st["file"], "wb") as fh:
        for seed, frames, actions in eps[: st["episodes"]]:
            t = st["start_step"]
            fh.write(np.ascontiguousarray(frames[t - reach:t]).astype(np.uint8).tobytes())      # (reach, S, S, 3) RGB
            fh.write(actions[t - reach:t - 1].astype(np.int32).tobytes())                       # (reach - 1,)
    # a trace of dataset.History on dummy frames, so the page can check its own context buffer against it:
    # start with `reach` frames with ids 0..reach-1 and actions j % 9, then push frames; record which frame ids and
    # actions each context holds
    ids = torch.arange(reach, dtype=torch.float32)[:, None, None, None, None].expand(reach, 1, 3, 1, 1).clone()
    h = History(ids, (torch.arange(reach - 1) % 9)[:, None], offsets)
    trace = []
    for j in range(st["history_check_steps"]):
        cx, ac = h.context(torch.tensor([(7 * j + 3) % 9]))
        trace.append({"frames": [int(v) for v in cx[0, ::3, 0, 0]], "actions": [int(v) for v in ac[0]]})
        h.push(torch.full((1, 3, 1, 1), float(reach + j)))
    # the same trace from a short history (the clamp at the start of an episode) and the key -> action table
    sys.path.insert(0, str(ROOT / "serve"))
    from server import keys_to_action
    ids = torch.arange(K, dtype=torch.float32)[:, None, None, None, None].expand(K, 1, 3, 1, 1).clone()
    h = History(ids, (torch.arange(K - 1) % 9)[:, None], offsets)
    trace_short = []
    for j in range(st["history_check_steps"]):
        cx, ac = h.context(torch.tensor([(5 * j + 1) % 9]))
        trace_short.append({"frames": [int(v) for v in cx[0, ::3, 0, 0]], "actions": [int(v) for v in ac[0]]})
        h.push(torch.full((1, 3, 1, 1), float(K + j)))
    keymap = [keys_to_action({"up": bool(b & 1), "down": bool(b & 2), "left": bool(b & 4), "right": bool(b & 8)}) for b in range(16)]
    json.dump({"history_check": trace, "history_check_short": trace_short, "keymap": keymap, "fps": cfg["fps"],
               "ort_version": cfg["ort_version"], "n": st["episodes"], "reach": reach, "size": S, "offsets": [int(o) for o in offsets], "ctx_sigma": cfg["ctx_sigma"],
               "episodes": [e[0] for e in eps[: st["episodes"]]], "start_step": st["start_step"],
               "layout": "per start: frames u8 (reach,S,S,3) then actions i32 (reach-1,)"}, open(out / "starts.json", "w"), indent=1)
    print(f"wrote {out}/cases.bin, starts.bin and their json")


if __name__ == "__main__":
    main()
