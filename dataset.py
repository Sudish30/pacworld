"""Windowed next-frame dataset over the recorded Ms. Pac-Man episodes.

Builds a one-time cache of all frames resized to size x size, splits episodes
into train/val (val episode seeds frozen in configs/val_episodes.json), and
samples windows of (context frames, context actions, target frame).

Window at target index i inside one episode:
  context frames  = frames[i-K .. i-1]
  context actions = actions[i-K .. i-1]   (actions[i-1] is the action that produced frames[i])
  target frame    = frames[i]
Frames are returned as float tensors in [-1, 1], shaped (B, K*3, H, W) / (B, 3, H, W).

Run directly to build the cache and print split statistics:
  python dataset.py --seed 0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import load_config


def build_cache(cfg):
    d = cfg["data"]
    files = sorted(Path(d["folder"]).glob("ep_*.npz"))
    if not files:
        raise SystemExit(f"no episodes in {d['folder']}")
    size = d["size"]
    frames, actions, rewards, seeds, starts = [], [], [], [], [0]
    for k, f in enumerate(files):
        ep = np.load(f)
        small = np.stack([np.asarray(Image.fromarray(x).resize((size, size), Image.BOX)) for x in ep["frames"]])
        frames.append(small)
        # align actions with frames: actions[j] follows frames[j]; the final frame has no action (-1)
        actions.append(np.concatenate([ep["actions"], [-1]]).astype(np.int64))
        rewards.append(np.concatenate([ep["rewards"], [0.0]]).astype(np.float32))
        seeds.append(int(ep["seed"]))
        starts.append(starts[-1] + len(small))
        if (k + 1) % 50 == 0:
            print(f"  cached {k + 1}/{len(files)} episodes")
    out = Path(d["cache"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, frames=np.concatenate(frames), actions=np.concatenate(actions), rewards=np.concatenate(rewards),
             ep_seed=np.asarray(seeds, dtype=np.int64), ep_start=np.asarray(starts, dtype=np.int64))
    print(f"wrote {out}: {starts[-1]} frames from {len(files)} episodes at {size}x{size}")


def load_cache(cfg):
    path = Path(cfg["data"]["cache"])
    if not path.exists():
        print(f"cache {path} missing; building it")
        build_cache(cfg)
    c = np.load(path)
    return {k: c[k] for k in c.files}


def load_split(cfg, ep_seeds):
    """Return (train_episode_idx, val_episode_idx). Val seeds are frozen on first creation."""
    d = cfg["data"]
    path = Path(d["val_episodes"])
    if path.exists():
        val_seeds = set(json.load(open(path))["val_episode_seeds"])
    else:
        rng = np.random.default_rng(d["split_seed"])
        n_val = max(1, int(round(d["val_frac"] * len(ep_seeds))))
        val_seeds = set(int(s) for s in rng.choice(ep_seeds, n_val, replace=False))
        json.dump({"val_episode_seeds": sorted(val_seeds),
                   "note": "frozen validation episodes; never train on these"}, open(path, "w"), indent=2)
        print(f"wrote {path} with {n_val} val episodes")
    val_idx = [i for i, s in enumerate(ep_seeds) if int(s) in val_seeds]
    train_idx = [i for i, s in enumerate(ep_seeds) if int(s) not in val_seeds]
    missing = val_seeds - set(int(s) for s in ep_seeds)
    if missing:
        print(f"warning: {len(missing)} val episode seeds not found in cache")
    return train_idx, val_idx


class WindowDataset:
    """Samples (context, actions, target) windows from a fixed set of episodes."""

    def __init__(self, cache, episode_idx, context):
        self.K = context
        self.frames = torch.from_numpy(cache["frames"])          # (N, H, W, 3) uint8
        self.actions = torch.from_numpy(cache["actions"])        # (N,) int64
        starts = cache["ep_start"]
        targets = []
        for e in episode_idx:
            a, b = int(starts[e]), int(starts[e + 1])
            targets.append(np.arange(a + context, b))
        self.targets = torch.from_numpy(np.concatenate(targets)) if targets else torch.zeros(0, dtype=torch.long)
        self.offsets = torch.arange(-context, 0)

    def __len__(self):
        return len(self.targets)

    def get(self, target_idx):
        """target_idx: LongTensor (B,) of absolute frame indices. Returns CPU tensors."""
        ctx_idx = target_idx[:, None] + self.offsets                 # (B, K)
        ctx = self.frames[ctx_idx]                                   # (B, K, H, W, 3)
        B, K, H, W, C = ctx.shape
        ctx = ctx.permute(0, 1, 4, 2, 3).reshape(B, K * C, H, W)
        tgt = self.frames[target_idx].permute(0, 3, 1, 2)            # (B, 3, H, W)
        acts = self.actions[ctx_idx]                                 # (B, K)
        return to_float(ctx), acts, to_float(tgt)

    def sample(self, batch_size, generator):
        pick = torch.randint(len(self.targets), (batch_size,), generator=generator)
        return self.get(self.targets[pick])

    def fixed_batches(self, batch_size, n_batches, seed):
        g = torch.Generator().manual_seed(seed)
        pick = torch.randperm(len(self.targets), generator=g)[: batch_size * n_batches]
        return [self.targets[pick[i * batch_size:(i + 1) * batch_size]] for i in range(n_batches)]


def to_float(u8):
    return u8.float().div_(127.5).sub_(1.0)


def to_uint8(x):
    return ((x.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)


def get_datasets(cfg):
    cache = load_cache(cfg)
    train_idx, val_idx = load_split(cfg, cache["ep_seed"])
    K = cfg["data"]["context"]
    return WindowDataset(cache, train_idx, K), WindowDataset(cache, val_idx, K), cache, (train_idx, val_idx)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/model0.yaml")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--rebuild", action="store_true", help="rebuild the cache even if it exists")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.rebuild:
        build_cache(cfg)
    train, val, cache, (ti, vi) = get_datasets(cfg)
    print(f"episodes: {len(ti)} train / {len(vi)} val")
    print(f"windows:  {len(train)} train / {len(val)} val")
    g = torch.Generator().manual_seed(args.seed)
    ctx, acts, tgt = train.sample(4, g)
    print(f"sample: ctx {tuple(ctx.shape)} {ctx.dtype} in [{ctx.min():.1f},{ctx.max():.1f}], "
          f"acts {tuple(acts.shape)} {acts.tolist()[0]}, target {tuple(tgt.shape)}")
    assert (acts >= 0).all(), "context actions must never include the -1 end-of-episode marker"
