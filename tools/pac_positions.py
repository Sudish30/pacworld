"""Pac-Man's position in every frame of the cache, from the pixel detector (eval/detectors.py). Label-free.

Reads only the cached frames (and one native frame for the detector's maze reference) - no actions, no RAM. Writes
data.pac_positions: float32 (N, 2) = (row, col) in cache pixels, NaN where the detector finds no Pac-Man. Used by
arm B's loss weight (lam.player_weight), the label-free key map (tools/lam_keymap.py) and T_det (tools/lam_checks.py).

  python tools/pac_positions.py --config configs/lam-B.yaml --seed 0 [--workers 16]
"""
import argparse
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from common import load_config  # noqa: E402
from dataset import codec_of, load_cache  # noqa: E402
import detectors as D  # noqa: E402

_REF = None
_FRAMES = None
_CODEC = None


def _init(cfg):
    global _REF, _FRAMES, _CODEC
    _REF = D.load_reference(cfg)
    cache = load_cache(cfg, mmap=True)          # each worker maps the frames; nothing else of the cache is used
    _FRAMES, _CODEC = cache["frames"], codec_of(cache)


def _chunk(bounds):
    a, b = bounds
    out = np.full((b - a, 2), np.nan, np.float32)
    for k, f in enumerate(_CODEC.decode_np(_FRAMES[a:b])):
        p = _REF.sprites(f)["pac"]
        if p is not None:
            out[k] = p[:2]
    return a, out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--chunk", type=int, default=2000, help="frames per task")
    p.add_argument("--limit", type=int, help="only the first N frames (smoke tests); the output is then partial")
    a = p.parse_args()
    np.random.seed(a.seed)
    cfg = load_config(ROOT / a.config)
    d = cfg["data"]
    if d["cache"].endswith(".npz"):
        raise SystemExit("tools/pac_positions.py needs a streamed .npy cache (workers memory-map it); convert the .npz first")
    N = len(load_cache(cfg, mmap=True)["frames"])
    n = min(N, a.limit) if a.limit else N
    jobs = [(s, min(s + a.chunk, n)) for s in range(0, n, a.chunk)]
    pos = np.full((N, 2), np.nan, np.float32)
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
        for k, (s, out) in enumerate(pool.imap_unordered(_chunk, jobs), 1):
            pos[s:s + len(out)] = out
            if k % 50 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} chunks ({time.time() - t0:.0f}s)")
    out = ROOT / d["pac_positions"]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, pos)
    found = np.isfinite(pos[:n, 0]).mean()
    meta = {"frames": int(N), "processed": int(n), "found_frac": float(found), "cache": d["cache"],
            "sha256": hashlib.sha256(pos.tobytes()).hexdigest(), "seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(out.with_suffix(".json"), "w"), indent=2)
    print(f"wrote {out}: {n}/{N} frames, Pac-Man found in {100 * found:.2f}% ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
