"""Windowed next-frame dataset over the recorded Ms. Pac-Man episodes.

Builds a one-time cache of all frames resized to size x size, splits episodes
into train/val (val episode seeds frozen in configs/val_episodes.json), and
samples windows of (context frames, context actions, target frame).

Window at target index i inside one episode, for context offsets o_1 < ... < o_K = -1
(data.context_offsets; default -K .. -1, i.e. the K consecutive frames before the target):
  context frames  = frames[max(i + o_k, episode start)]
  context actions = actions[same indices]  (actions[i-1] is the action that produced frames[i])
  target frame    = frames[i]
Offsets that reach before the episode start are clamped to the episode's first frame.
gather_context() is the only implementation of this rule: WindowDataset (training) and
History (autoregressive rollouts in training evals, the eval harness and the demo server)
both call it, so a rollout sees exactly the windows the model was trained on.
Frames are returned as float tensors in [-1, 1], shaped (B, K*3, H, W) / (B, 3, H, W).

data.folder may be one folder or a list. A cache path ending in .npy is written as a
stream (frames in the .npy, everything else in <stem>_meta.npz) so that building it never
holds more than one chunk of episodes in memory; a .npz path keeps the original format.

Run directly to build the cache and print split statistics:
  python dataset.py --seed 0
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import load_config


def episode_files(d, root=Path(".")):
    """All recorded episodes of data.folder (one folder or a list), folder by folder, first copy of a seed wins."""
    folders = [d["folder"]] if isinstance(d["folder"], str) else d["folder"]
    files, seen = [], set()
    for folder in folders:
        for f in sorted((root / folder).glob("ep_*.npz")):
            if f.stem in seen:
                print(f"warning: {f} duplicates an episode seed from an earlier folder; skipped")
                continue
            seen.add(f.stem)
            files.append(f)
    return files


def episode_path(d, seed, root=Path(".")):
    folders = [d["folder"]] if isinstance(d["folder"], str) else d["folder"]
    for folder in folders:
        f = root / folder / f"ep_{seed}.npz"
        if f.exists():
            return f
    raise FileNotFoundError(f"ep_{seed}.npz not in {folders}")


def _episode_len(path):
    with np.load(path) as ep:
        return len(ep["actions"]) + 1


def _load_small(job):
    path, size = job
    with np.load(path) as ep:
        small = np.stack([np.asarray(Image.fromarray(x).resize((size, size), Image.BOX)) for x in ep["frames"]])
        # align actions with frames: actions[j] follows frames[j]; the final frame has no action (-1)
        actions = np.concatenate([ep["actions"], [-1]]).astype(np.int64)
        rewards = np.concatenate([ep["rewards"], [0.0]]).astype(np.float32)
        return small, actions, rewards, int(ep["seed"])


def meta_path(cache_path):
    return cache_path.with_name(cache_path.stem + "_meta.npz")


def build_cache(cfg, workers, chunk):
    d = cfg["data"]
    files = episode_files(d)
    if not files:
        raise SystemExit(f"no episodes in {d['folder']}")
    size, out = d["size"], Path(d["cache"])
    out.parent.mkdir(parents=True, exist_ok=True)
    stream = out.suffix == ".npy"
    frames, actions, rewards, seeds = [], [], [], []
    with Pool(workers) as pool:
        lengths = pool.map(_episode_len, files, chunksize=16)
        starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        fp = None
        if stream:   # header first, then each episode's bytes in order: sequential writes, one chunk in memory
            fp = open(out, "wb")
            np.lib.format.write_array_header_2_0(fp, {"descr": "|u1", "fortran_order": False,
                                                      "shape": (int(starts[-1]), size, size, 3)})
        for c0 in range(0, len(files), chunk):
            for k, (small, a, r, seed) in enumerate(pool.map(_load_small, [(f, size) for f in files[c0:c0 + chunk]]), c0):
                assert len(small) == lengths[k], f"{files[k]} changed while caching"
                if stream:
                    fp.write(np.ascontiguousarray(small).tobytes())
                else:
                    frames.append(small)
                actions.append(a)
                rewards.append(r)
                seeds.append(seed)
            print(f"  cached {min(c0 + chunk, len(files))}/{len(files)} episodes")
        if stream:
            fp.close()
    meta = dict(actions=np.concatenate(actions), rewards=np.concatenate(rewards),
                ep_seed=np.asarray(seeds, dtype=np.int64), ep_start=starts)
    if stream:
        np.savez(meta_path(out), **meta)
    else:
        np.savez(out, frames=np.concatenate(frames), **meta)
    print(f"wrote {out}: {starts[-1]} frames from {len(files)} episodes at {size}x{size}")


def load_cache(cfg, mmap=False):
    """Cache as a dict of arrays. mmap=True maps the frames of a .npy cache instead of reading them into RAM."""
    path = Path(cfg["data"]["cache"])
    if not path.exists():
        raise SystemExit(f"cache {path} missing; build it with: python dataset.py --config <config> --seed 0 --rebuild")
    if path.suffix == ".npy":
        c = dict(np.load(meta_path(path)))
        c["frames"] = np.load(path, mmap_mode="r" if mmap else None)
        return c
    c = np.load(path)
    return {k: c[k] for k in c.files}


def load_split(cfg, ep_seeds):
    """Return (train_episode_idx, val_episode_idx). Val seeds are frozen on first creation."""
    d = cfg["data"]
    path = Path(d["val_episodes"])
    if path.exists():
        val_seeds = set(json.load(open(path))["val_episode_seeds"])
    elif "val_base" in d:
        raise SystemExit(f"{path} missing; freeze it once with: python tools/freeze_val_split.py --config <config> --seed {d['split_seed']}")
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


def context_offsets(d):
    """Context frame offsets relative to the target: data.context_offsets, or the K consecutive frames -K .. -1."""
    K = d["context"]
    offsets = list(d.get("context_offsets") or range(-K, 0))
    if len(offsets) != K or offsets[-1] != -1 or any(a >= b for a, b in zip(offsets, offsets[1:])):
        raise SystemExit(f"data.context_offsets must be {K} strictly increasing offsets ending in -1, got {offsets}")
    return offsets


def gather_context(frames, actions, target_idx, first_idx, offsets):
    """The context rule. Every training window and every rollout step goes through this function.

    frames / actions: time-first tensors with actions[j] = the action taken after frames[j].
    target_idx (B,): index of the frame to predict. first_idx (B,): index of the first frame of the
    target's episode (or of a rollout history). offsets (K,): negative offsets from context_offsets().
    Context position k of target i is index max(i + o_k, first_idx): an offset that reaches before the
    episode start reads the episode's first frame, together with that frame's action, and never the
    previous episode. Returns (frames[idx], actions[idx], idx) with idx shaped (B, K).
    """
    idx = torch.maximum(target_idx[:, None] + offsets[None, :], first_idx[:, None])
    return frames[idx], actions[idx], idx


class WindowDataset:
    """Samples (context, actions, target) windows from a fixed set of episodes."""

    def __init__(self, cache, episode_idx, offsets):
        self.offsets = torch.as_tensor(offsets, dtype=torch.long)
        self.K = len(offsets)
        self.frames = torch.from_numpy(cache["frames"])          # (N, H, W, 3) uint8
        self.actions = torch.from_numpy(cache["actions"])        # (N,) int64
        self.ep_start = torch.from_numpy(cache["ep_start"])      # (E + 1,) int64
        targets = []
        for e in episode_idx:
            a, b = int(self.ep_start[e]), int(self.ep_start[e + 1])
            targets.append(np.arange(a + self.K, b))             # a target needs K real frames before it
        self.targets = torch.from_numpy(np.concatenate(targets)) if targets else torch.zeros(0, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def first_idx(self, target_idx):
        return self.ep_start[torch.searchsorted(self.ep_start, target_idx, right=True) - 1]

    def get(self, target_idx):
        """target_idx: LongTensor (B,) of absolute frame indices. Returns CPU tensors."""
        ctx, acts, _ = gather_context(self.frames, self.actions, target_idx, self.first_idx(target_idx), self.offsets)
        B, K, H, W, C = ctx.shape                                    # (B, K, H, W, 3)
        ctx = ctx.permute(0, 1, 4, 2, 3).reshape(B, K * C, H, W)
        tgt = self.frames[target_idx].permute(0, 3, 1, 2)            # (B, 3, H, W)
        return to_float(ctx), acts, to_float(tgt)

    def sample(self, batch_size, generator):
        pick = torch.randint(len(self.targets), (batch_size,), generator=generator)
        return self.get(self.targets[pick])

    def set_event_targets(self, index, kinds, radius):
        """Targets within `radius` steps of an event (one pool per kind), restricted to this dataset's own targets
        and to the event's episode. index: npz from tools/build_event_index.py."""
        own = set(self.targets.tolist())
        self.event_pools = []
        for kind in kinds:
            ev, ep = index[kind], index[kind + "_ep"]
            lo = np.maximum(ev - radius, self.ep_start.numpy()[ep])
            hi = np.minimum(ev + radius, self.ep_start.numpy()[ep + 1] - 1)
            t = np.unique(np.concatenate([np.arange(a, b + 1) for a, b in zip(lo, hi)])) if len(ev) else np.zeros(0, np.int64)
            t = np.array([x for x in t.tolist() if x in own], dtype=np.int64)
            if not len(t):
                raise SystemExit(f"no event targets of kind {kind} in this split")
            self.event_pools.append(torch.from_numpy(t))
        return [len(t) for t in self.event_pools]

    def sample_mixed(self, batch_size, generator, frac):
        """A batch with round(frac * batch_size) event windows, split evenly over the event kinds; the rest uniform."""
        n_ev = int(round(frac * batch_size))
        per = [n_ev // len(self.event_pools) + (1 if i < n_ev % len(self.event_pools) else 0) for i in range(len(self.event_pools))]
        picks = [pool[torch.randint(len(pool), (n,), generator=generator)] for pool, n in zip(self.event_pools, per)]
        picks.append(self.targets[torch.randint(len(self.targets), (batch_size - n_ev,), generator=generator)])
        return self.get(torch.cat(picks))

    def fixed_batches(self, batch_size, n_batches, seed):
        g = torch.Generator().manual_seed(seed)
        pick = torch.randperm(len(self.targets), generator=g)[: batch_size * n_batches]
        return [self.targets[pick[i * batch_size:(i + 1) * batch_size]] for i in range(n_batches)]


class History:
    """Frames and actions of B parallel rollouts, newest last; context() is gather_context() on this buffer.

    Only the last `reach` frames are kept. The buffer must start at the episode's first frame or already
    hold `reach` frames (from_episodes guarantees this), so an offset is clamped here exactly when it is
    clamped in training: when it reaches before the first frame of the episode.
    """

    def __init__(self, frames, actions, offsets):
        """frames (T, B, 3, H, W) float in [-1, 1]; actions (T - 1, B): actions[j] was taken after frames[j]."""
        self.offsets = torch.as_tensor(offsets, dtype=torch.long)
        self.reach = -int(self.offsets.min())
        pending = torch.zeros_like(actions[:1]) if len(actions) else torch.zeros((1, frames.shape[1]), dtype=torch.long, device=frames.device)
        self.frames = frames[-self.reach:]
        self.actions = torch.cat([actions, pending])[-self.reach:]   # last slot: the action context() is given

    @classmethod
    def from_episodes(cls, episodes, start, offsets, device):
        """episodes: list of (frames uint8 (T, H, W, 3), actions (T,)) arrays, each from its episode's first frame.
        The first predicted frame is index `start`; the buffer holds frames[max(0, start - reach) : start]."""
        lo = max(0, start + min(offsets))
        frames = torch.stack([to_float(torch.from_numpy(np.ascontiguousarray(f[lo:start])).permute(0, 3, 1, 2)) for f, _ in episodes], 1)
        actions = torch.stack([torch.from_numpy(np.asarray(a[lo:start - 1], dtype=np.int64)) for _, a in episodes], 1)
        return cls(frames.to(device), actions.to(device), offsets)

    def context(self, action):
        """action (B,): the action taken after the newest frame. Returns ctx (B, K*3, H, W) and actions (B, K)."""
        T = len(self.frames)
        if T < len(self.offsets):
            raise ValueError(f"history holds {T} frames; training targets always have {len(self.offsets)} real frames before them")
        self.actions[-1] = action
        f, a, _ = gather_context(self.frames, self.actions, torch.tensor([T]), torch.tensor([0]), self.offsets)
        return f[0].transpose(0, 1).flatten(1, 2), a[0].T            # (K, B, 3, H, W) -> (B, K*3, H, W)

    def push(self, frame):
        """Append the frame that followed the last context() call (a prediction, or the real frame when teacher forcing)."""
        self.frames = torch.cat([self.frames, frame[None].to(self.frames.dtype)])[-self.reach:]
        self.actions = torch.cat([self.actions, torch.zeros_like(self.actions[:1])])[-self.reach:]


def to_float(u8):
    return u8.float().div_(127.5).sub_(1.0)


def to_uint8(x):
    return ((x.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)


def get_datasets(cfg):
    cache = load_cache(cfg)
    train_idx, val_idx = load_split(cfg, cache["ep_seed"])
    offsets = context_offsets(cfg["data"])
    return WindowDataset(cache, train_idx, offsets), WindowDataset(cache, val_idx, offsets), cache, (train_idx, val_idx)


def check_windows(cfg, train, val, cache, n_episodes, seed):
    """Exhaustive checks of the context rule on this cache (run before trusting a new context layout)."""
    offsets, starts = train.offsets, train.ep_start
    # 1. no window of any train or val target touches another episode, and no context action is the -1 end marker
    for name, ds in (("train", train), ("val", val)):
        clamped = 0
        for t in ds.targets.split(1 << 18):
            # the indices gather_context itself used (actions stand in for frames: both are read at the same idx)
            _, acts, idx = gather_context(ds.actions, ds.actions, t, ds.first_idx(t), offsets)
            ep_of = lambda i: torch.searchsorted(starts, i, right=True) - 1
            assert (ep_of(idx) == ep_of(t)[:, None]).all(), f"{name}: a context index left its episode"
            assert (idx < t[:, None]).all() and (acts >= 0).all(), f"{name}: bad context index or action"
            clamped += int((idx[:, 0] != t + offsets[0]).sum())
        print(f"  {name}: {len(ds.targets)} windows stay inside their episode; {clamped} "
              f"({100 * clamped / max(len(ds.targets), 1):.1f}%) have at least one offset clamped to the episode start")
    # 2. a History fed an episode from its first frames yields, at every step, exactly the dataset's window
    rng = np.random.default_rng(seed)
    n_steps = 0
    for e in rng.choice(len(starts) - 1, n_episodes, replace=False):
        a, b = int(starts[e]), int(starts[e + 1])
        ep = (cache["frames"][a:b], cache["actions"][a:b])
        for start in sorted({train.K, train.K + 1, -int(offsets.min()) - 1, -int(offsets.min()) + 3, 100}):
            if start < train.K or start >= b - a - 1:
                continue
            h = History.from_episodes([ep], start, offsets.tolist(), "cpu")
            for i in range(start, min(start + 120, b - a)):
                ctx, acts = h.context(torch.tensor([int(ep[1][i - 1])]))
                want_ctx, want_acts, want_tgt = train.get(torch.tensor([a + i]))
                assert torch.equal(ctx, want_ctx) and torch.equal(acts, want_acts), f"history != dataset at episode {e} step {i}"
                h.push(want_tgt)                                          # teacher-forced: push the real frame
                n_steps += 1
    print(f"  history buffer == dataset window on {n_steps} steps of {n_episodes} episodes "
          f"(rollouts started at steps {train.K}..100, so clamped and unclamped steps are both covered)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/model0.yaml")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--rebuild", action="store_true", help="rebuild the cache even if it exists")
    p.add_argument("--workers", type=int, default=16, help="processes used to build the cache")
    p.add_argument("--chunk", type=int, default=64, help="episodes held in memory at once while building")
    p.add_argument("--check-windows", type=int, metavar="N", help="verify the context rule on all windows and N random episodes")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.rebuild or not Path(cfg["data"]["cache"]).exists():
        build_cache(cfg, args.workers, args.chunk)
    train, val, cache, (ti, vi) = get_datasets(cfg)
    print(f"episodes: {len(ti)} train / {len(vi)} val")
    print(f"windows:  {len(train)} train / {len(val)} val")
    g = torch.Generator().manual_seed(args.seed)
    ctx, acts, tgt = train.sample(4, g)
    print(f"sample: ctx {tuple(ctx.shape)} {ctx.dtype} in [{ctx.min():.1f},{ctx.max():.1f}], "
          f"acts {tuple(acts.shape)} {acts.tolist()[0]}, target {tuple(tgt.shape)}")
    assert (acts >= 0).all(), "context actions must never include the -1 end-of-episode marker"
    print(f"context offsets: {train.offsets.tolist()}")
    if args.check_windows:
        check_windows(cfg, train, val, cache, args.check_windows, args.seed)
