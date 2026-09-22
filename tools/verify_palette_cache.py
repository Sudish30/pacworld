"""Round-trip check of a palette cache: decode random cached frames and compare them, pixel for pixel, with the
same frames resized again from the raw recordings. (The builder already maps every pixel of every frame; this is
an independent check that the stored indices, the palette and the frame order all line up.)

  python tools/verify_palette_cache.py --config configs/m1-2M-128-ctx6s16.yaml --seed 0 --frames 2000
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import load_config  # noqa: E402
from dataset import FrameCodec, episode_files, load_cache, resize_frames  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--frames", type=int, default=2000)
    a = p.parse_args()
    cfg = load_config(a.config)
    d = cfg["data"]
    cache = load_cache(cfg, mmap=True)
    if "palette" not in cache:
        raise SystemExit("not a palette cache")
    codec, starts, files = FrameCodec(cache["palette"]), cache["ep_start"], episode_files(d, ROOT)
    assert len(files) == len(starts) - 1, "episode count differs from the cache"
    rng = np.random.default_rng(a.seed)
    picks = np.sort(rng.choice(int(starts[-1]), a.frames, replace=False))
    ep_of = np.searchsorted(starts, picks, side="right") - 1
    bad = checked = 0
    for e in np.unique(ep_of):
        raw = np.load(files[e])
        assert int(raw["seed"]) == int(cache["ep_seed"][e]), f"episode {e}: seed mismatch"
        local = picks[ep_of == e] - starts[e]
        want = resize_frames(raw["frames"][local], d["size"], d.get("resample", "box"))
        got = codec.decode_np(cache["frames"][picks[ep_of == e]])
        bad += int((want != got).any(axis=(1, 2, 3)).sum())
        checked += len(local)
    rgb = int(starts[-1]) * d["size"] * d["size"] * 3
    print(f"round trip: {checked} random frames from {len(np.unique(ep_of))} episodes, {checked - bad} identical to the raw "
          f"recording resized again, {bad} differ -> {'LOSSLESS' if bad == 0 else 'NOT LOSSLESS'}")
    print(f"size: {cache['frames'].nbytes / 2**30:.1f} GiB as palette indices ({len(cache['palette'])} colours) vs "
          f"{rgb / 2**30:.1f} GiB as RGB")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
