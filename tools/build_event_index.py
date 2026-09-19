"""Index rare game events in the cached dataset, from the recorded RAM.

Writes events.index (npz) with absolute cache frame indices, one array per event kind:
  fright_end    first frame after a frightened phase (RAM frightened timer goes from > 0 to 0)
  pen_release   first frame a ghost is outside the pen after being inside (RAM ghost positions)
and ep_of_event arrays so that sampling windows never leave the event's episode.
Episode order and offsets come from the cache metadata, so indices match the cache exactly.

  python tools/build_event_index.py --config configs/m1-2M-ctx6s16-ft-events.yaml --seed 0
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from common import load_config  # noqa: E402
from dataset import episode_path, load_cache  # noqa: E402
import detectors as D  # noqa: E402
from pen_timer_analysis import in_pen  # noqa: E402

_CFG = None


def _init(cfg):
    global _CFG
    _CFG = cfg


def _events(seed):
    ram = np.load(episode_path(_CFG["data"], int(seed), ROOT))["ram"]
    fr = ram[:, _CFG["events"]["frightened_ram"]] > 0
    fright_end = np.nonzero(fr[:-1] & ~fr[1:])[0] + 1
    rel = []
    for g in D.GHOST_NAMES:
        ip = in_pen(ram, g)
        rel.append(np.nonzero(ip[:-1] & ~ip[1:])[0] + 1)
    return len(ram), fright_end, np.unique(np.concatenate(rel))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    cfg = load_config(a.config)
    cache = load_cache(cfg, mmap=True)
    seeds, starts = cache["ep_seed"], cache["ep_start"]
    with Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
        res = pool.map(_events, list(seeds), chunksize=8)
    out = {"fright_end": [], "pen_release": [], "fright_end_ep": [], "pen_release_ep": []}
    for e, (n, fe, pr) in enumerate(res):
        assert n == starts[e + 1] - starts[e], f"episode {seeds[e]}: RAM length {n} != cached length {starts[e + 1] - starts[e]}"
        out["fright_end"].append(starts[e] + fe)
        out["fright_end_ep"].append(np.full(len(fe), e))
        out["pen_release"].append(starts[e] + pr)
        out["pen_release_ep"].append(np.full(len(pr), e))
    out = {k: np.concatenate(v).astype(np.int64) for k, v in out.items()}
    path = ROOT / cfg["events"]["index"]
    np.savez(path, **out)
    E = len(seeds)
    print(f"wrote {path}: {len(out['fright_end'])} frightened-phase ends ({len(out['fright_end']) / E:.2f} per episode), "
          f"{len(out['pen_release'])} pen releases ({len(out['pen_release']) / E:.2f} per episode) in {E} episodes / {starts[-1]} frames")


if __name__ == "__main__":
    main()
