"""Reported diagnostic for the browser demo (not a gate): does the exported half-precision graph keep Pac-Man on
screen as often as PyTorch over a long rollout? Rolls out web/model_fp16.onnx (ONNX Runtime, CPU) and the PyTorch
model from the demo's starting histories with the same actions and independent noise, and reports the share of frames
in which the pixel detector finds Pac-Man. It checks the exported graph and 16-bit weights, not the WebGPU kernels
(those are covered by web/bench.html's single-frame and same-noise rollout checks).

  python tools/web_rollout_check.py --config configs/web_demo.yaml --seed 0
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
sys.path.insert(0, str(ROOT / "tools"))
from common import load_config  # noqa: E402
from dataset import History, context_offsets, to_uint8  # noqa: E402
from model1 import build_model, euler_sample  # noqa: E402
from export_onnx import longest_val_episodes  # noqa: E402
import detectors as D  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/web_demo.yaml")
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    cfg = load_config(ROOT / a.config)
    pc = cfg["presence_check"]
    import onnxruntime as ort
    sess = ort.InferenceSession(str(ROOT / cfg["out_dir"] / cfg["model_fp16"]), providers=["CPUExecutionProvider"])
    ck = torch.load(ROOT / cfg["checkpoint"], map_location="cpu")
    model = build_model(ck["cfg"])
    model.load_state_dict(ck["ema"])
    model.eval()
    offsets = context_offsets(ck["cfg"]["data"])
    ref = D.load_reference(load_config(ROOT / pc["detector_config"]))
    S, res = cfg["data"]["size"], {"onnx_fp16": [], "pytorch": []}
    sig = torch.full((1,), float(cfg["ctx_sigma"]))
    for e, (seed, frames, actions) in enumerate(longest_val_episodes(cfg, cfg["starts"]["episodes"])):
        for name in res:
            gen = torch.Generator().manual_seed(a.seed * 1000 + e * 10 + (name == "pytorch"))
            hist = History.from_episodes([(frames, actions)], cfg["starts"]["start_step"], offsets, "cpu")
            found = 0
            for j in range(pc["steps"]):
                action = int(pc["actions"][(j // pc["hold"]) % len(pc["actions"])])
                ctx, act = hist.context(torch.tensor([action]))
                ctx = ctx + cfg["ctx_sigma"] * torch.randn(ctx.shape, generator=gen)
                if name == "pytorch":
                    with torch.no_grad():
                        pred = euler_sample(model, ctx, act, sig, cfg["sampler_steps"], ck["cfg"]["diffusion"], gen)
                else:
                    noise = torch.randn(1, 3, S, S, generator=gen)
                    pred = torch.from_numpy(sess.run(None, {"ctx": ctx.numpy(), "actions": act.numpy(), "noise": noise.numpy()})[0])
                hist.push(pred)
                found += ref.sprites(to_uint8(pred[0]).permute(1, 2, 0).numpy())["pac"] is not None
            res[name].append(found / pc["steps"])
            print(f"episode {seed} {name}: Pac-Man found in {found}/{pc['steps']} frames", flush=True)
    out = {k: {"per_start": v, "mean": float(np.mean(v))} for k, v in res.items()}
    json.dump(out, open(ROOT / cfg["out_dir"] / "presence_check.json", "w"), indent=1)
    print(f"Pac-Man present: ONNX fp16 {out['onnx_fp16']['mean']:.3f}, PyTorch {out['pytorch']['mean']:.3f} "
          f"({cfg['starts']['episodes']} starts x {pc['steps']} steps)")


if __name__ == "__main__":
    main()
