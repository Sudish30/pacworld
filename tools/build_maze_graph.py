"""Mine a walkable-maze graph from Pac-Man's recorded RAM positions (for the pellet-seeking recorder).

Nodes are RAM positions (x = ram[10], y = ram[16]); a directed edge joins the positions of two consecutive
steps within one life (short moves only, so respawn jumps and tunnel wraps are left out).

  python tools/build_maze_graph.py --config configs/record_pellet.yaml --seed 0
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import load_config  # noqa: E402

_S = None


def _init(s):
    global _S
    _S = s


def _edges(path):
    ram = np.load(path)["ram"]
    x, y, lives = ram[:, _S["pac_x_ram"]].astype(int), ram[:, _S["pac_y_ram"]].astype(int), ram[:, _S["lives_ram"]].astype(int)
    d = np.abs(np.diff(x)) + np.abs(np.diff(y))
    ok = (np.diff(lives) == 0) & (d > 0) & (d <= _S["max_step"])
    return np.stack([x[:-1][ok], y[:-1][ok], x[1:][ok], y[1:][ok]], 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    s = load_config(a.config)["seek"]
    files = [f for folder in s["graph_from"] for f in sorted((ROOT / folder).glob("ep_*.npz"))]
    with Pool(a.workers, initializer=_init, initargs=(s,)) as pool:
        e = np.concatenate(pool.map(_edges, files, chunksize=16))
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    keep = counts >= s["min_edge_count"]
    out = ROOT / s["graph"]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, edges=uniq[keep].astype(np.int16), counts=counts[keep])
    nodes = {tuple(r[:2]) for r in uniq[keep]} | {tuple(r[2:]) for r in uniq[keep]}
    print(f"wrote {out}: {keep.sum()} edges, {len(nodes)} positions from {len(files)} episodes")
    for name, (px, py) in s["power_pellets"].items():
        near = [n for n in nodes if abs(n[0] - px) + abs(n[1] - py) <= s["pellet_radius"]]
        print(f"  power pellet {name} at RAM ({px}, {py}): {len(near)} graph positions within {s['pellet_radius']}")


if __name__ == "__main__":
    main()
