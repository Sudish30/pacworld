"""The README's demo GIF: the real game beside two models' rollouts of the same held-out episode, same start and
recorded actions, from the frames eval_rollouts.py --save-all-preds stored. No model is run here.

The rollout is chosen by a fixed rule (configs/pen_stats.yaml, demo_gif.pick): the first saved rollout in which the
`pick.run` model keeps the pen occupied for at least `pick.min_longest` steps. It is an example of the failure, not a
random sample; the rates are in the README table.

  python tools/make_demo_gif.py --config configs/pen_stats.yaml --seed 0
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from common import load_config  # noqa: E402
from dataset import episode_path  # noqa: E402
import detectors as D  # noqa: E402
from pen_bootstrap import run_longest  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/pen_stats.yaml")
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    c = yaml.safe_load(open(ROOT / a.config))
    g = c["demo_gif"]
    pick = c["runs"][g["pick"]["run"]]
    ecfg = load_config(ROOT / pick["config"])
    _, ml, _ = run_longest(ecfg, ROOT / pick["dir"], g["steps"])
    rows = np.nonzero(ml >= g["pick"]["min_longest"])[0]
    if not len(rows):
        raise SystemExit("no rollout matches demo_gif.pick")
    row = int(rows[0])
    job = list(csv.DictReader(open(ROOT / pick["dir"] / "jobs.csv")))[row]
    start = ecfg["rollout"]["start_step"]
    native = np.load(episode_path(ecfg["data"], int(job["episode"]), ROOT))["frames"][start:start + g["steps"]]
    panels = [("real game", D.downsample_frames(native, *D.frame_geometry(ecfg)))]
    for name, label in g["panels"].items():
        r = c["runs"][name]
        j = list(csv.DictReader(open(ROOT / r["dir"] / "jobs.csv")))[row]
        if (j["episode"], j["seed"]) != (job["episode"], job["seed"]):
            raise SystemExit(f"{name}: jobs.csv row {row} is a different rollout")
        panels.append((label, np.load(ROOT / r["dir"] / "preds_all.npy", mmap_mode="r")[row, :g["steps"]]))
    T = min(len(f) for _, f in panels)
    s, size, pad, top = g["upscale"], panels[0][1].shape[1] * g["upscale"], 4, 16
    font = ImageFont.load_default(size=11)
    frames = []
    for t in range(0, T, g["frame_stride"]):
        sheet = Image.new("RGB", (len(panels) * (size + pad) - pad, top + size + 12), "black")
        draw = ImageDraw.Draw(sheet)
        for k, (label, f) in enumerate(panels):
            x = k * (size + pad)
            draw.text((x + 2, 2), label, fill="white", font=font)
            sheet.paste(Image.fromarray(np.asarray(f[t]).astype(np.uint8)).resize((size, size), Image.NEAREST), (x, top))
        draw.text((2, top + size), f"step {t + 1}", fill=(160, 160, 160), font=font)
        frames.append(sheet.quantize(256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE))
    out = ROOT / g["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=round(1000 * g["frame_stride"] / g["fps"]), loop=0, optimize=True)
    print(f"rollout row {row}: episode {job['episode']}, sampler seed {job['seed']}; {g['pick']['run']} longest pen stay "
          f"{int(ml[row])} steps; {len(frames)} frames -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
