"""Freeze the validation split of an extended dataset, once.

The new split is every episode of the existing frozen split (data.val_base, reused as-is)
plus data.val_frac of the episodes in data.val_new_folder, drawn once with --seed and
written to data.val_episodes. The file is never overwritten: a frozen split stays frozen.

  python tools/freeze_val_split.py --config configs/m1-2M-ctx4.yaml --seed 0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import load_config  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True, help="seed of the one-time draw; must equal data.split_seed")
    args = p.parse_args()
    d = load_config(args.config)["data"]
    if args.seed != d["split_seed"]:
        raise SystemExit(f"--seed {args.seed} differs from data.split_seed {d['split_seed']}")
    out = ROOT / d["val_episodes"]
    if out.exists():
        raise SystemExit(f"{out} already exists; the split is frozen and is not redrawn")

    base = sorted(json.load(open(ROOT / d["val_base"]))["val_episode_seeds"])
    new = sorted(int(f.stem.split("_")[1]) for f in (ROOT / d["val_new_folder"]).glob("ep_*.npz"))
    clash = sorted(set(new) & set(base))
    if clash:   # such an episode is dropped from the cache as a duplicate, and it already is a val episode
        print(f"note: {len(clash)} new episode seeds equal an original val seed: {clash}")
    pool = [s for s in new if s not in set(base)]
    n_val = int(round(d["val_frac"] * len(pool)))
    rng = np.random.default_rng(args.seed)
    picked = sorted(int(s) for s in rng.choice(pool, n_val, replace=False))
    json.dump({"val_episode_seeds": sorted(base + picked),
               "from_val_base": base, "from_val_new_folder": picked,
               "note": f"frozen validation episodes; never train on these. {len(base)} reused from {d['val_base']} + "
                       f"{n_val} of the {len(pool)} episodes in {d['val_new_folder']} (val_frac {d['val_frac']}, seed {args.seed})"},
              open(out, "w"), indent=2)
    print(f"wrote {out}: {len(base)} original + {n_val} of {len(pool)} new = {len(base) + n_val} val episodes")


if __name__ == "__main__":
    main()
