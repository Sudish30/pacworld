"""Colour/structure detectors for downsampled Ms. Pac-Man frames (64x64 or 128x128).

Everything is derived from how the dataset cache builds a frame from the 172x160
maze crop. With area (BOX) downsampling every pixel is a linear mix of the native
palette colours; with NEAREST every pixel IS a native palette colour, so the same
unmixing still applies with coverage 0 or 1. data.resample picks which, and the
reference maze is built the same way, so detector and frames always agree.

  MazeReference   static maze at 64x64 built once from a native frame: wall
                  reference mask, per-pellet footprints, pellet-free background,
                  and each pellet's additive contribution to the background.
  pellet_presence per-pellet present/absent from the frame's "pinkness" over the
                  pellet's footprint (validated ~98% against native truth).
  wall_mask       pixels reading as wall, outside pellet footprints.
  sprites         Pac-Man, the 4 ghosts and frightened ghosts by background
                  subtraction: residual against the expected background (maze +
                  currently present pellets), per-pixel two-colour unmixing to
                  pick the sprite colour, largest blob -> coverage-weighted centroid.

Run directly to validate against native-resolution truth on recorded episodes:
  python eval/detectors.py --seed 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import episode_files  # noqa: E402
from tools.ghost_visibility import GHOSTS, FRIGHTENED, BACKGROUND_BLUE  # noqa: E402

BG = (0, 28, 136)
WALL = (228, 111, 111)
PAC = (210, 164, 74)
BLACK = (0, 0, 0)
SPRITES = {"pac": PAC, **GHOSTS, "frightened": FRIGHTENED}
GHOST_NAMES = list(GHOSTS)


def pooled_variant(col):
    """Colour of a sprite drawn in only some of the 4 max-pooled frames: element-wise max with the background."""
    return tuple(max(c, b) for c, b in zip(col, BG))


# every sprite colour and its max-pooled variant, with the class each belongs to
VARIANT_COLS, VARIANT_CLASS = [], []
for _name, _col in SPRITES.items():
    for _v in dict.fromkeys([_col, pooled_variant(_col)]):
        VARIANT_COLS.append(_v)
        VARIANT_CLASS.append(_name)
VARIANT_COLS = np.array(VARIANT_COLS, float)
NATIVE_H, NATIVE_W = 172, 160
PELLET_MAX_PX = 12        # native pixels; dots are 4x2 = 8
POWER_MAX_PX = 40         # power pellets are larger blobs, still far smaller than any wall piece


# ----------------------------------------------------------------------------- linear area downsample
def _area_matrix(n_in, n_out):
    """(n_out, n_in) matrix of overlap fractions for exact area-averaging resize."""
    M = np.zeros((n_out, n_in))
    scale = n_in / n_out
    for o in range(n_out):
        a, b = o * scale, (o + 1) * scale
        for i in range(int(np.floor(a)), int(np.ceil(b))):
            M[o, i] = max(0.0, min(b, i + 1) - max(a, i))
    return M / scale


def _nearest_matrix(n_in, n_out):
    """(n_out, n_in) 0/1 selection matrix, read out of PIL itself so it matches the cache exactly."""
    from PIL import Image
    probe = np.repeat(np.arange(n_in, dtype=np.uint8)[:, None], 2, axis=1)
    pick = np.asarray(Image.fromarray(probe, mode="L").resize((2, n_out), Image.NEAREST))[:, 0].astype(int)
    M = np.zeros((n_out, n_in))
    M[np.arange(n_out), pick] = 1.0
    return M


class Downsampler:
    def __init__(self, size=64, mode="box"):
        f = _area_matrix if mode == "box" else _nearest_matrix
        self.Mh = f(NATIVE_H, size)
        self.Mw = f(NATIVE_W, size)

    def __call__(self, img):
        """img (H, W) or (H, W, C) float -> (size, size[, C])."""
        if img.ndim == 2:
            return self.Mh @ img @ self.Mw.T
        return np.stack([self.Mh @ img[..., c] @ self.Mw.T for c in range(img.shape[-1])], axis=-1)


def _components(mask, conn8=True):
    """Label connected components; returns (labels, sizes)."""
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    sizes = []
    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)] + ([(1, 1), (1, -1), (-1, 1), (-1, -1)] if conn8 else [])
    n = 0
    for i, j in zip(*np.nonzero(mask)):
        if lab[i, j]:
            continue
        n += 1
        lab[i, j] = n
        stack, sz = [(i, j)], 0
        while stack:
            a, b = stack.pop()
            sz += 1
            for da, db in nbrs:
                p, q = a + da, b + db
                if 0 <= p < H and 0 <= q < W and mask[p, q] and not lab[p, q]:
                    lab[p, q] = n
                    stack.append((p, q))
        sizes.append(sz)
    return lab, np.array(sizes)


def _exact(frame, col, allow_bg_blue=False):
    m = (frame[..., 0] == col[0]) & (frame[..., 1] == col[1])
    return m & ((frame[..., 2] == col[2]) | (frame[..., 2] == BACKGROUND_BLUE)) if allow_bg_blue else m & (frame[..., 2] == col[2])


# ----------------------------------------------------------------------------- maze reference
class MazeReference:
    def __init__(self, native_frame, cfg, size=64, mode="box"):
        d = cfg["detector"]
        self.cfg = d
        self.size = size
        self.mode = mode
        self.down = Downsampler(size, mode)
        bg, wall = np.array(BG, float), np.array(WALL, float)
        self.bg_col, self.wall_col = bg, wall
        self.pink_axis = (wall - bg) / ((wall - bg) @ (wall - bg))

        wallpix = _exact(native_frame, WALL)
        lab, sizes = _components(wallpix, conn8=False)
        pellet_ids = [i + 1 for i, s in enumerate(sizes) if s <= POWER_MAX_PX]
        pellet_native = np.isin(lab, pellet_ids)
        wall_native = wallpix & ~pellet_native
        self.n_dots = int(sum(1 for i, s in enumerate(sizes) if s <= PELLET_MAX_PX))
        self.n_power = len(pellet_ids) - self.n_dots

        # pellet-free, sprite-free maze
        maze = np.zeros_like(native_frame, dtype=float) + bg
        maze[wall_native] = wall
        maze[_exact(native_frame, BLACK)] = 0.0
        # downsample the reference exactly as the dataset cache does (PIL BOX on uint8)
        self.bg64 = downsample_frames(maze.astype(np.uint8)[None], size, mode)[0].astype(float)   # (size, size, 3)
        self.wall_ref = self.pinkness(self.bg64) > d["wall_pinkness"]   # same test the frame mask uses

        # per-pellet footprints and additive background contributions
        self.pellets = []
        delta_col = wall - bg
        for pid in pellet_ids:
            m = (lab == pid).astype(float)
            cov = self.down(m)
            cells = np.argwhere(cov > 0.02)
            w = cov[cells[:, 0], cells[:, 1]]
            delta = cov[cells[:, 0], cells[:, 1], None] * delta_col[None]     # colour added to those cells
            self.pellets.append({"cells": cells, "w": w / w.sum(), "delta": delta,
                                 "ref": float((self.pinkness(self.bg64 + self._paint(cells, delta))[cells[:, 0], cells[:, 1]] * w / w.sum()).sum())})
        self.pellet_cells = np.zeros((size, size), bool)
        for p in self.pellets:
            self.pellet_cells[p["cells"][:, 0], p["cells"][:, 1]] = True
        self.wall_eval = ~self.pellet_cells                               # cells where wall IoU is evaluated

    def _paint(self, cells, delta):
        img = np.zeros((self.size, self.size, 3))
        img[cells[:, 0], cells[:, 1]] += delta
        return img

    def pinkness(self, frame64):
        return np.clip(((frame64.astype(float) - self.bg_col) * self.pink_axis).sum(-1), 0, 1)

    def pellet_presence(self, frame64):
        pk = self.pinkness(frame64)
        frac = self.cfg["pellet_present_frac"]
        return np.array([(pk[p["cells"][:, 0], p["cells"][:, 1]] * p["w"]).sum() >= frac * p["ref"] for p in self.pellets])

    def expected_background(self, presence):
        img = self.bg64.copy()
        for p, on in zip(self.pellets, presence):
            if on:
                img[p["cells"][:, 0], p["cells"][:, 1]] += p["delta"]
        return img

    def wall_mask(self, frame64):
        return (self.pinkness(frame64) > self.cfg["wall_pinkness"]) & self.wall_eval

    def wall_iou(self, frame64):
        m, r = self.wall_mask(frame64), self.wall_ref & self.wall_eval
        return (m & r).sum() / max((m | r).sum(), 1)

    def unmix(self, frame64, presence=None):
        """Per-pixel sprite unmixing against every sprite colour variant.

        Returns (accept, best_class, best_alpha, magnitude): a boolean mask of pixels that
        read as sprite rather than background, the sprite class each belongs to, its blend
        coverage, and the residual magnitude.
        """
        d = self.cfg
        f = frame64.astype(float)
        if presence is None:
            presence = self.pellet_presence(frame64)
        exp = self.expected_background(presence)
        resid = f - exp
        mag = np.linalg.norm(resid, axis=-1)
        diff = VARIANT_COLS[None, None] - exp[:, :, None, :]               # (64, 64, V, 3)
        alpha = np.clip((resid[:, :, None, :] * diff).sum(-1) / np.maximum((diff * diff).sum(-1), 1e-6), 0, 1)
        fit = np.linalg.norm(resid[:, :, None, :] - alpha[..., None] * diff, axis=-1)
        best = fit.argmin(-1)
        best_alpha = np.take_along_axis(alpha, best[..., None], -1)[..., 0]
        accept = (mag > d["residual_threshold"]) & (fit.min(-1) < d["fit_threshold"]) & (best_alpha >= d["min_alpha"])
        return accept, np.array(VARIANT_CLASS, dtype=object)[best], best_alpha, mag

    def ghost_mass(self, frame64, presence=None):
        """Alpha-weighted coverage of ghost-coloured pixels: dict per ghost plus 'total'.

        Unlike sprites(), this does not require a blob to survive the size test, so it
        measures how much ghost colour is on screen even when it has smeared apart.
        """
        accept, cls, alpha, _ = self.unmix(frame64, presence)
        out = {g: float(alpha[accept & (cls == g)].sum()) for g in GHOST_NAMES}
        out["frightened"] = float(alpha[accept & (cls == "frightened")].sum())
        out["total"] = float(sum(out[g] for g in GHOST_NAMES))
        return out

    def sprites(self, frame64, presence=None):
        """dict name -> (row, col, weight) or None, for pac, red, pink, cyan, orange, frightened.

        Residual against the expected background; per-pixel two-colour unmixing gives a
        class-agnostic sprite mask; each connected blob is then assigned by an
        alpha-weighted vote of its pixels' labels, splitting only where two sprites overlap.
        """
        d = self.cfg
        accept, best_class, best_alpha, mag = self.unmix(frame64, presence)
        out = {name: None for name in SPRITES}
        if not accept.any():
            return out
        lab, sizes = _components(accept)
        classes = list(SPRITES)
        for k in range(1, len(sizes) + 1):
            mm = lab == k
            # alpha-weighted vote of the per-pixel unmixing labels within this blob
            votes = {c: float(best_alpha[mm & (best_class == c)].sum()) for c in classes}
            strong = [c for c, v in votes.items() if v >= d["min_blob_weight"]]
            if not strong:
                if sum(votes.values()) >= d["min_blob_weight"]:
                    strong = [max(votes, key=votes.get)]      # fragmented labels: dominant class takes the blob
                else:
                    continue
            for c in strong:
                sel = mm & (best_class == c) if len(strong) > 1 else mm     # split only when two sprites overlap
                w = best_alpha[sel]
                weight = float(w.sum())
                if weight < d["min_blob_weight"] or (out[c] is not None and out[c][2] >= weight):
                    continue
                r, cc = np.nonzero(sel)
                out[c] = (float((r * w).sum() / weight), float((cc * w).sum() / weight), weight)
        return out


def _frightened_blobs(self, frame64, presence=None):
    """Every frightened (blue) ghost in the frame: list of (row, col, weight), one per connected blue blob.

    sprites() keeps only the heaviest blob of a class, which is right for the four coloured ghosts (one each)
    but not for frightened ghosts, which all share one colour. Two overlapping blue ghosts form one blob; callers
    that need a count divide the weight by a single ghost's typical weight.
    """
    accept, best_class, best_alpha, _ = self.unmix(frame64, presence)
    mask = accept & (best_class == "frightened")
    out = []
    if mask.any():
        lab, sizes = _components(mask)
        for k in range(1, len(sizes) + 1):
            mm = lab == k
            w = best_alpha[mm]
            weight = float(w.sum())
            if weight >= self.cfg["min_blob_weight"]:
                r, c = np.nonzero(mm)
                out.append((float((r * w).sum() / weight), float((c * w).sum() / weight), weight))
    return out


MazeReference.frightened_blobs = _frightened_blobs


def native_sprite_centroid(native_frame, col, min_px=10, size=64):
    m = _exact(native_frame, col, allow_bg_blue=True)
    if m.sum() < min_px:
        return None
    r, c = np.nonzero(m)
    return np.array([r.mean() * size / NATIVE_H, c.mean() * size / NATIVE_W])


def downsample_frames(native_frames, size=64, mode="box"):
    """Exactly the resize dataset.build_cache uses -> uint8 (N, size, size, 3)."""
    from dataset import resize_frames
    return resize_frames(native_frames, size, mode)


def frame_geometry(cfg):
    """(size, resample mode) the frames of this config are built with."""
    d = cfg["data"]
    return d["size"], d.get("resample", "box")


def load_reference(cfg):
    """Build the maze reference from the first frame of the first recorded episode."""
    first = episode_files(cfg["data"], ROOT)[0]
    size, mode = frame_geometry(cfg)
    return MazeReference(np.load(first)["frames"][0], cfg, size, mode)


# ----------------------------------------------------------------------------- validation
def validate(cfg, n_episodes=3, stride=6, seed=0):
    from common import load_config  # noqa
    ref = load_reference(cfg)
    scale = ref.size / 64        # position errors are in cache pixels; report them per 64px-equivalent too
    print(f"maze reference: {ref.size}x{ref.size} '{ref.mode}', {ref.n_dots} dots + {ref.n_power} power pellets, "
          f"wall cells {ref.wall_ref.sum()}")
    files = episode_files(cfg["data"], ROOT)
    rng = np.random.default_rng(seed)
    files = [files[i] for i in rng.choice(len(files), n_episodes, replace=False)]
    stats = {n: {"tot": 0, "hit": 0, "absent": 0, "false": 0, "err": []} for n in ["pac"] + GHOST_NAMES}
    pellet_agree, pellet_iou, wall_iou = [], [], []
    for f in files:
        ep = np.load(f)
        native, small = ep["frames"], downsample_frames(ep["frames"], ref.size, ref.mode)
        pellet_native_masks = None
        for t in range(0, len(native), stride):
            x, s = native[t], small[t]
            presence = ref.pellet_presence(s)
            wallpix = _exact(x, WALL)
            # native pellet truth: pellet present if most of its native pixels are still wall-coloured
            if pellet_native_masks is None:
                lab, sizes = _components(_exact(native[0], WALL), conn8=False)
                pellet_native_masks = [lab == (i + 1) for i, sz in enumerate(sizes) if sz <= POWER_MAX_PX]
            gt = np.array([wallpix[m].mean() > 0.5 for m in pellet_native_masks])
            pellet_agree.append((gt == presence).mean())
            pellet_iou.append((gt & presence).sum() / max((gt | presence).sum(), 1))
            wall_iou.append(ref.wall_iou(s))
            det = ref.sprites(s, presence)
            for n in stats:
                truth = native_sprite_centroid(x, SPRITES[n], size=ref.size)
                if truth is None:
                    stats[n]["absent"] += 1
                    stats[n]["false"] += det[n] is not None
                    continue
                stats[n]["tot"] += 1
                if det[n] is None:
                    continue
                stats[n]["hit"] += 1
                stats[n]["err"].append(float(np.hypot(det[n][0] - truth[0], det[n][1] - truth[1])))
    print(f"pellets: per-pellet agreement {np.mean(pellet_agree):.4f} (min {np.min(pellet_agree):.4f}), set IoU {np.mean(pellet_iou):.4f}")
    print(f"walls:   IoU on real frames {np.mean(wall_iou):.4f} (min {np.min(wall_iou):.4f})")
    ok = True
    for n, s in stats.items():
        e = np.array(s["err"]) if s["err"] else np.array([np.nan])
        rate = s["hit"] / max(s["tot"], 1)
        print(f"{n:7s} detected {s['hit']}/{s['tot']} ({100 * rate:.1f}%)  false pos {s['false']}/{s['absent']}  "
              f"pos err: mean {np.nanmean(e):.2f} px = {np.nanmean(e) / scale:.2f} at 64px-equivalent,  p90 {np.nanpercentile(e, 90):.2f}  "
              f"max {np.nanmax(e):.1f}  (>{2 * scale:.0f}px: {100 * np.mean(e > 2 * scale):.1f}%)")
        ok &= rate >= 0.95 and np.nanmean(e) < scale
    print("VALIDATION", "PASS" if ok else "FAIL", "(target: >=95% detection, <1 px mean error per sprite at 64px-equivalent)")


if __name__ == "__main__":
    from common import load_config
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--episodes", type=int, default=3)
    a = p.parse_args()
    validate(load_config(a.config), a.episodes, seed=a.seed)
