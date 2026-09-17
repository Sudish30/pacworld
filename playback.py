"""Render a recorded episode as a GIF and optionally verify frame/action alignment.

--check-alignment re-simulates the episode from its stored seed, replays the
stored actions, and asserts that every frame and RAM snapshot matches exactly.
"""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import crop, get_ram, load_config, make_env


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/record.yaml")
    p.add_argument("--seed", type=int, required=True, help="used to pick an episode when --episode is omitted")
    p.add_argument("--episode", help="path to an ep_*.npz; default: random file from data/{mode}")
    p.add_argument("--mode", choices=["agent", "random"], help="which data/ subfolder to pick from")
    p.add_argument("--out", help="output GIF path; default: next to the episode")
    p.add_argument("--max-frames", type=int, help="only render the first N frames")
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--check-alignment", action="store_true")
    return p.parse_args()


def pick_episode(args, cfg):
    if args.episode:
        return Path(args.episode)
    folder = Path(cfg["out_dir"]) / (args.mode or cfg["mode"])
    files = sorted(folder.glob("ep_*.npz"))
    if not files:
        raise SystemExit(f"no episodes in {folder}")
    return files[np.random.default_rng(args.seed).integers(len(files))]


def check_alignment(ep, cfg):
    env = make_env(cfg)
    seed = int(ep["seed"])
    obs, _ = env.reset(seed=seed)
    frames, actions, ram = ep["frames"], ep["actions"], ep["ram"]
    assert np.array_equal(crop(obs, cfg), frames[0]), "initial frame mismatch"
    assert np.array_equal(get_ram(env), ram[0]), "initial RAM mismatch"
    for i, a in enumerate(actions):
        obs, reward, terminated, truncated, _ = env.step(int(a))
        if not np.array_equal(crop(obs, cfg), frames[i + 1]):
            raise AssertionError(f"frame mismatch at step {i}: frames[{i+1}] does not follow actions[{i}]")
        if not np.array_equal(get_ram(env), ram[i + 1]):
            raise AssertionError(f"RAM mismatch at step {i}")
        if reward != ep["rewards"][i]:
            raise AssertionError(f"reward mismatch at step {i}: {reward} vs {ep['rewards'][i]}")
        if terminated != ep["terminated"][i]:
            raise AssertionError(f"terminated mismatch at step {i}")
    env.close()
    print(f"alignment OK: seed={seed}, {len(actions)} steps, {len(frames)} frames, "
          f"every frames[i+1] follows from actions[i] (frames, RAM, rewards, terminated all match)")


def render_gif(ep, cfg, out, max_frames=None):
    pb = cfg["playback"]
    scale, band = pb["scale"], pb["band_px"] * pb["scale"]
    frames, actions, rewards = ep["frames"], ep["actions"], ep["rewards"]
    names = [str(n) for n in ep["action_meanings"]]
    n = len(frames) if max_frames is None else min(max_frames, len(frames))
    font = ImageFont.load_default(size=int(4 * scale))
    h, w = frames.shape[1] * scale, frames.shape[2] * scale
    out_frames = []
    for i in range(n):
        img = Image.fromarray(frames[i]).resize((w, h), Image.NEAREST)
        canvas = Image.new("RGB", (w, h + band), "black")
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        if i < len(actions):
            text = f"t={i:<4d} {names[actions[i]]:<9s} r={rewards[i]:.0f}"
        else:
            text = f"t={i:<4d} (final frame)"
        draw.text((2 * scale, h + scale), text, fill="white", font=font)
        out_frames.append(np.asarray(canvas))
    imageio.mimsave(out, out_frames, duration=1000 / pb["fps"], loop=0)
    print(f"wrote {out} ({n} frames, {w}x{h + band})")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    path = pick_episode(args, cfg)
    ep = np.load(path)
    print(f"episode {path}: {len(ep['actions'])} steps, frames {ep['frames'].shape}, "
          f"return {ep['rewards'].sum():.0f}, seed {int(ep['seed'])}")
    if args.check_alignment:
        check_alignment(ep, cfg)
    if not args.no_gif:
        out = args.out or path.with_suffix(".gif")
        render_gif(ep, cfg, out, args.max_frames)


if __name__ == "__main__":
    main()
