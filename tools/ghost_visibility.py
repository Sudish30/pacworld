"""Dataset-quality check: how often are all four ghosts visible in recorded frames?

Ms. Pac-Man on the Atari 2600 flickers ghosts across emulator frames, so a
recorder that does not max-pool enough raw frames produces observations with
missing ghosts. This tool counts, per ghost colour, the pixels in each frame
and reports the fraction of frames in which all four ghosts are present.
It reports three numbers:
  * all-four fraction over all frames (confounded by legitimate game phases:
    ghosts stacked in the pen at the start of each life, death animations,
    frightened/blue ghosts, and sprites that overlap and blend colours);
  * all-four fraction over "in-play" frames (start-of-life, death-animation
    and frightened frames excluded), a fairer view of the same thing;
  * transient dropout rate per ghost: absent at t but present at t-1 and
    t+1. This isolates flicker, which is what max-pooling is meant to fix.
It also prints episode/return statistics and can save an example frame.

Usage:
  python tools/ghost_visibility.py --seed 0 --folder data/agent
  python tools/ghost_visibility.py --seed 0 --folder data/agent --save-frame data/agent/four_ghosts.png
"""
import argparse
import glob
import itertools
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_config  # noqa: E402

# Sprite palette of the level-1 maze. The blue channel can also be 136 when a
# max-pooled sprite overlaps the (0, 28, 136) background of the other frame.
GHOSTS = {"red": (200, 72, 72), "pink": (198, 89, 179), "cyan": (84, 184, 153), "orange": (180, 122, 48)}
BACKGROUND_BLUE = 136
FRIGHTENED = (66, 114, 194)      # all ghosts turn this blue after a power pellet
LIVES_RAM = 123                  # ram[123] = lives remaining
START_OF_LIFE_STEPS = 90         # ghosts sit stacked in the pen for roughly this long
DEATH_ANIM_STEPS = 12            # ghosts vanish while Ms. Pac-Man's death animation plays, before lives decrements


def ghost_masks(frame):
    out = {}
    for name, (r, g, b) in GHOSTS.items():
        out[name] = (frame[..., 0] == r) & (frame[..., 1] == g) & ((frame[..., 2] == b) | (frame[..., 2] == BACKGROUND_BLUE))
    return out


def ghost_counts(frame):
    return {k: int(m.sum()) for k, m in ghost_masks(frame).items()}


def all_four(frame, min_pixels):
    return all(v >= min_pixels for v in ghost_counts(frame).values())


def centroids(frame, min_pixels):
    out = {}
    for name, m in ghost_masks(frame).items():
        if m.sum() < min_pixels:
            return None
        ys, xs = np.nonzero(m)
        out[name] = (float(ys.mean()), float(xs.mean()))
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/record.yaml")
    p.add_argument("--seed", type=int, required=True, help="seed for episode subsampling")
    p.add_argument("--folder", help="episode folder; default: {out_dir}/{mode} from config")
    p.add_argument("--sample-episodes", type=int, default=30, help="episodes to scan for visibility (0 = all)")
    p.add_argument("--min-pixels", type=int, default=15, help="pixels of a ghost colour needed to count it as visible")
    p.add_argument("--min-frac", type=float, help="exit non-zero if the gated metric is below this fraction")
    p.add_argument("--gate", choices=["all", "in-play"], default="in-play", help="which all-four fraction --min-frac applies to")
    p.add_argument("--max-dropout", type=float, help="exit non-zero if any ghost's transient dropout rate exceeds this fraction")
    p.add_argument("--save-frame", help="save a mid-game frame with four well-separated ghosts to this PNG")
    p.add_argument("--scale", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    folder = Path(args.folder) if args.folder else Path(cfg["out_dir"]) / cfg["mode"]
    files = sorted(folder.glob("ep_*.npz"))
    if not files:
        raise SystemExit(f"no episodes in {folder}")

    lens, rets = [], []
    for f in files:
        ep = np.load(f)
        lens.append(len(ep["actions"]))
        rets.append(float(ep["rewards"].sum()))
    print(f"{folder}: {len(files)} episodes, {sum(lens)} steps")
    print(f"  episode length min/mean/max = {min(lens)}/{np.mean(lens):.0f}/{max(lens)}")
    print(f"  return         min/mean/max = {min(rets):.0f}/{np.mean(rets):.0f}/{max(rets):.0f}")

    rng = np.random.default_rng(args.seed)
    n = len(files) if args.sample_episodes == 0 else min(args.sample_episodes, len(files))
    sample = [files[i] for i in sorted(rng.choice(len(files), n, replace=False))]
    hist = np.zeros(5, dtype=int)
    inplay_total = inplay_four = 0
    dropout = np.zeros(len(GHOSTS))
    dropout_base = np.zeros(len(GHOSTS))
    for f in sample:
        ep = np.load(f)
        frames, lives = ep["frames"], ep["ram"][:, LIVES_RAM].astype(int)
        vis = np.array([[v >= args.min_pixels for v in ghost_counts(x).values()] for x in frames])
        blue = np.array([((x[..., 0] == FRIGHTENED[0]) & (x[..., 1] == FRIGHTENED[1]) & (x[..., 2] == FRIGHTENED[2])).any() for x in frames])
        k = vis.sum(1)
        hist += np.bincount(k, minlength=5)
        in_play = ~blue
        in_play[:START_OF_LIFE_STEPS] = False
        for t in np.nonzero(np.diff(lives) < 0)[0]:
            in_play[max(0, t - DEATH_ANIM_STEPS):t + START_OF_LIFE_STEPS] = False
        in_play[-DEATH_ANIM_STEPS:] = False  # final death animation before game over
        inplay_total += in_play.sum()
        inplay_four += (k[in_play] == 4).sum()
        neighbours = vis[:-2] & vis[2:]
        dropout += (neighbours & ~vis[1:-1]).sum(0)
        dropout_base += neighbours.sum(0)
    total = hist.sum()
    frac_all, frac_inplay = hist[4] / total, inplay_four / max(inplay_total, 1)
    drop_rates = dropout / np.maximum(dropout_base, 1)
    print(f"  ghost visibility over {n} episodes / {total} frames (>= {args.min_pixels} px per ghost colour):")
    for i in range(5):
        print(f"    {i} ghosts visible: {100 * hist[i] / total:5.1f}%")
    print(f"  ALL FOUR VISIBLE, all frames:     {hist[4]}/{total} = {100 * frac_all:.1f}%")
    print(f"  ALL FOUR VISIBLE, in-play frames: {inplay_four}/{inplay_total} = {100 * frac_inplay:.1f}%"
          f"  (excludes first {START_OF_LIFE_STEPS} steps of each life, {DEATH_ANIM_STEPS} steps of each death animation, and frightened frames)")
    print("  transient dropout per ghost (absent at t, present at t-1 and t+1):")
    for name, d, b, r in zip(GHOSTS, dropout, dropout_base, drop_rates):
        print(f"    {name:7s} {100 * r:5.2f}%  ({int(d)}/{int(b)})")
    frac = frac_inplay if args.gate == "in-play" else frac_all

    if args.save_frame:
        saved = False
        for f in sample:
            frames = np.load(f)["frames"]
            for i in range(250, len(frames)):
                c = centroids(frames[i], 25)
                if c and all(np.hypot(*np.subtract(c[a], c[b])) > 20 for a, b in itertools.combinations(c, 2)):
                    h, w = frames.shape[1:3]
                    Image.fromarray(frames[i]).resize((w * args.scale, h * args.scale), Image.NEAREST).save(args.save_frame)
                    print(f"  saved {args.save_frame}: {f.name} step {i}, " + ", ".join(f"{k}=({v[0]:.0f},{v[1]:.0f})" for k, v in c.items()))
                    saved = True
                    break
            if saved:
                break
        if not saved:
            print("  no mid-game frame with four well-separated ghosts found")

    failed = False
    if args.min_frac is not None and frac < args.min_frac:
        print(f"FAIL: all-four fraction ({args.gate}) {100 * frac:.1f}% is below {100 * args.min_frac:.0f}%")
        failed = True
    if args.max_dropout is not None and drop_rates.max() > args.max_dropout:
        print(f"FAIL: max transient dropout {100 * drop_rates.max():.2f}% exceeds {100 * args.max_dropout:.2f}%")
        failed = True
    if failed:
        sys.exit(1)
    if args.min_frac is not None or args.max_dropout is not None:
        print("PASS")


if __name__ == "__main__":
    main()
