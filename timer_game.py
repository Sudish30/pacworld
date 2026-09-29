"""The hidden-timer game (notes/timer_rule_design.md): a ghost is penned for exactly N frames with no on-screen clock.

16x16 RGB, exact palette, no actions. A player sprite and the ghost random-walk (with inertia) in the roaming area;
after D ~ Uniform{roam_min..roam_max} steps the ghost is captured: the capture frame c is the first frame showing it
in the pen, it stays for frames c .. c+N-1 and reappears below the pen at frame c+N (the release). Nothing on screen
changes with the time spent in the pen, so the capture frame is the only clock.

Also here: the pixel detector (ghost penned / released), rollout starts, and the ideal observer limited to a context
layout - the best any model with those offsets can do, estimated from training episodes.
"""
import hashlib

import numpy as np

DIRS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])


def _walk(pos, d, rng, g):
    """One random-walk step of a 2x2 sprite with inertia, bounded to the roaming area."""
    r0, r1 = g["roam_rows"]
    c0, c1 = g["roam_cols"]
    if rng.random() >= g["keep_direction"]:
        d = int(rng.integers(4))
    for _ in range(8):
        nxt = pos + DIRS[d]
        if r0 <= nxt[0] <= r1 and c0 <= nxt[1] <= c1:
            return nxt, d
        d = int(rng.integers(4))
    return pos, d


def background(g):
    s = g["size"]
    img = np.empty((s, s, 3), np.uint8)
    img[:] = g["colors"]["background"]
    wall = g["colors"]["wall"]
    img[0, :] = img[-1, :] = img[:, 0] = img[:, -1] = wall
    r0, c0, r1, c1 = g["pen_box"]
    img[r0, c0:c1 + 1] = img[r1, c0:c1 + 1] = img[r0:r1 + 1, c0] = img[r0:r1 + 1, c1] = wall
    return img


def generate_episode(g, N, seed, render=True):
    """frames (L, S, S, 3) uint8 (None if not render), penned (L,) bool, captures (list of c)."""
    rng = np.random.default_rng(seed)
    L = g["episode_len"]
    bg = background(g)
    r0, r1 = g["roam_rows"]
    c0, c1 = g["roam_cols"]
    player = np.array([rng.integers(r0, r1 + 1), rng.integers(c0, c1 + 1)])
    ghost = np.array([rng.integers(r0, r1 + 1), rng.integers(c0, c1 + 1)])
    pd, gd = int(rng.integers(4)), int(rng.integers(4))
    roam_left = int(rng.integers(g["roam_min"], g["roam_max"] + 1))
    pen_left = 0                                  # frames still to spend in the pen (0 = roaming)
    frames = np.empty((L, g["size"], g["size"], 3), np.uint8) if render else None
    penned = np.zeros(L, bool)
    captures = []
    pen_cell, rel_cell = np.array(g["pen_cell"]), np.array(g["release_cell"])
    for t in range(L):
        if t > 0:
            player, pd = _walk(player, pd, rng, g)
            if pen_left > 0:
                pen_left -= 1
                if pen_left == 0:                 # release frame
                    ghost = rel_cell.copy()
                    roam_left = int(rng.integers(g["roam_min"], g["roam_max"] + 1))
            else:
                roam_left -= 1
                if roam_left <= 0:                # capture frame
                    ghost = pen_cell.copy()
                    pen_left = N
                    captures.append(t)
                else:
                    ghost, gd = _walk(ghost, gd, rng, g)
        penned[t] = pen_left > 0
        if render:
            f = bg.copy()
            f[player[0]:player[0] + 2, player[1]:player[1] + 2] = g["colors"]["player"]
            f[ghost[0]:ghost[0] + 2, ghost[1]:ghost[1] + 2] = g["colors"]["ghost"]
            frames[t] = f
    return frames, penned, captures


def generate_split(g, N, split, render=True):
    """All episodes of a split: list of (frames, penned, captures, seed) and a sha256 over frames + penned."""
    n, s0 = g[f"n_{split}"], g[f"{split}_seed0"]
    eps = [(*generate_episode(g, N, s0 + e, render), s0 + e) for e in range(n)]
    h = hashlib.sha256()
    for fr, pen, _, _ in eps:
        if fr is not None:
            h.update(fr.tobytes())
        h.update(pen.tobytes())
    return eps, h.hexdigest()


# ----------------------------------------------------------------------------- detector
def ghost_state(frames, g):
    """(T, S, S, 3) uint8 -> (in_pen (T,), released (T,)) by nearest-palette classification."""
    cols = np.array(list(g["colors"].values()), float)
    ghost_k = list(g["colors"]).index("ghost")
    f = np.asarray(frames, float)
    lab = np.argmin(((f[..., None, :] - cols) ** 2).sum(-1), -1)
    red = lab == ghost_k
    r, c = g["pen_cell"]
    in_pen = red[:, r:r + 2, c:c + 2].sum((1, 2)) >= 2
    outside = red.copy()
    r0, c0, r1, c1 = g["pen_box"]
    outside[:, r0:r1 + 1, c0:c1 + 1] = False
    released = (outside.sum((1, 2)) >= 2) & ~in_pen
    return in_pen, released


def release_lag(released_after_capture):
    """Index k (1-based, frames c+1 ..) of the first released frame, or None (parked)."""
    hit = np.nonzero(released_after_capture)[0]
    return int(hit[0]) + 1 if len(hit) else None


def rollout_starts(eps, N, e):
    """(episode index, capture c) pairs usable as rollout starts: enough real history and room for the horizon."""
    horizon = e["horizon_mult"] * N + e["horizon_add"]
    out = []
    for i, (fr, pen, caps, _) in enumerate(eps):
        L = len(pen)
        ok = [c for c in caps if c >= e["min_history"] and c + horizon < L]
        out += [(i, c) for c in ok[: e["starts_per_episode"]]]
    return out, horizon


# ----------------------------------------------------------------------------- ideal observer
def _pattern(pen_seq, idx, first, offsets):
    return tuple(bool(pen_seq[max(idx + o, first)]) for o in offsets)


def observer_hazard(train_eps, offsets):
    """P(released at t | ghost state at the offsets around t), for every t whose previous frame is penned."""
    n, r = {}, {}
    for _, pen, _, _ in train_eps:
        for t in range(1, len(pen)):
            if not pen[t - 1]:
                continue
            p = _pattern(pen, t, 0, offsets)
            n[p] = n.get(p, 0) + 1
            if not pen[t]:
                r[p] = r.get(p, 0) + 1
    return {p: r.get(p, 0) / n[p] for p in n}


def observer_rollouts(hazard, pen_real, c, horizon, offsets, draws, rng):
    """Release lags of the ideal observer from capture c (real history before c+1, its own penned frames after).
    Returns (lags array with -1 = parked, number of unseen patterns met)."""
    lags = np.full(draws, -1, np.int64)
    unseen = 0
    seq = np.zeros(c + 1 + horizon, bool)
    seq[: c + 1] = pen_real[: c + 1]
    seq[c + 1:] = True                            # the observer's own frames stay penned until it releases
    haz = np.empty(horizon)
    for k in range(1, horizon + 1):
        p = _pattern(seq, c + k, 0, offsets)
        if p not in hazard:
            unseen += 1
        haz[k - 1] = hazard.get(p, 0.0)
    u = rng.random((draws, horizon))
    fire = u < haz[None]
    any_ = fire.any(1)
    lags[any_] = fire[any_].argmax(1) + 1
    return lags, unseen
