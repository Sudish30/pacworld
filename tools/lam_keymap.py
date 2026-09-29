"""Label-free key -> code map and resolver thresholds (design section 7.1-7.2). Frames only: no actions, no RAM.

Directions: for every code c, Pac-Man's displacement over the code's own transition (pixel detector positions,
tools/pac_positions.py) on TRAIN transitions. Moves of at least keymap.min_move_px vote for UP / RIGHT / DOWN / LEFT;
dir(c) = the winning direction if its share (purity) is >= keymap.purity, else "NONE" (an environment code). A code
whose transitions miss Pac-Man in keymap.undetected_env_frac of cases is also "NONE". L1 = every direction has a code.

Thresholds (needs the prior, train_lam.py --part prior): tau_legal = the keymap.tau_legal_percentile percentile of
p(true code | context) over keymap.calib_transitions random train transitions; tau_env = keymap.tau_env.
Writes <checkpoint_dir>/keymap.json, which lam.resolve() reads.

  python tools/lam_keymap.py --config configs/lam-A.yaml --seed 0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import load_config  # noqa: E402
from dataset import get_datasets  # noqa: E402
from lam import DIRECTIONS, Prior  # noqa: E402
from train_model0 import pick_device  # noqa: E402


def move_directions(pac, j, min_move, max_move):
    """Direction index (0 UP, 1 RIGHT, 2 DOWN, 3 LEFT) of Pac-Man's move frames[j] -> frames[j+1] for an index array j;
    -1 = no move (below min_move), -2 = undetected in either frame or a jump (tunnel wrap / respawn)."""
    d = pac[j + 1] - pac[j]                                  # (row, col) in cache pixels; rows grow downwards
    dr, dc = d[:, 0], d[:, 1]
    mag = np.hypot(dr, dc)
    out = np.where(np.abs(dr) >= np.abs(dc), np.where(dr < 0, 0, 2), np.where(dc > 0, 1, 3)).astype(np.int64)
    out[mag < min_move] = -1
    out[~np.isfinite(mag) | (mag > max_move)] = -2
    return out


def code_directions(codes, pac, transitions, n_codes, k):
    """dir(c) for every code from the given transition indices j (codes[j] is the code of j -> j+1)."""
    mv = move_directions(pac, transitions, k["min_move_px"], k["max_move_px"])
    c = codes[transitions]
    table, dirs = [], []
    for code in range(n_codes):
        m = mv[c == code]
        n = len(m)
        moves = m[m >= 0]
        votes = np.bincount(moves, minlength=4)
        purity = float(votes.max() / len(moves)) if len(moves) else 0.0
        undetected = float((m == -2).mean()) if n else 1.0
        d = DIRECTIONS[int(votes.argmax())] if len(moves) and purity >= k["purity"] and undetected < k["undetected_env_frac"] else "NONE"
        dirs.append(d)
        table.append({"code": code, "transitions": int(n), "share": float(n / max(len(c), 1)), "dir": d, "purity": purity,
                      "votes": dict(zip(DIRECTIONS, votes.tolist())), "still_frac": float((m == -1).mean()) if n else 0.0,
                      "undetected_frac": undetected})
    return dirs, table


def train_transitions(cache, train_idx):
    """Every transition j -> j+1 inside the train episodes (j on each episode's last frame excluded)."""
    s = cache["ep_start"]
    return np.concatenate([np.arange(s[e], s[e + 1] - 1) for e in train_idx])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--calib-transitions", type=int, help="override keymap.calib_transitions (smoke tests only)")
    a = p.parse_args()
    cfg = load_config(ROOT / a.config)
    k, n = cfg["keymap"], cfg["lam"]["n_codes"]
    if a.calib_transitions:
        k["calib_transitions"] = a.calib_transitions
    cfg["data"]["action_source"] = "lam"                     # the action slots hold codes; the labels are deleted
    train, _, cache, (train_idx, _) = get_datasets(cfg)
    codes, pac = cache["actions"], np.load(ROOT / cfg["data"]["pac_positions"])
    dirs, table = code_directions(codes, pac, train_transitions(cache, train_idx), n, k)
    missing = [d for d in DIRECTIONS if d not in dirs]

    device = pick_device(cfg["train"]["device"])
    ck = torch.load(ROOT / cfg["train"]["checkpoint_dir"] / "prior.pt", map_location=device)
    prior = Prior(ck["cfg"]).to(device).eval()
    prior.load_state_dict(ck["model"])
    from dataset import FrameCodec
    codec = FrameCodec(cache.get("palette")).to(device)
    g = torch.Generator().manual_seed(a.seed)
    pick = train.targets[torch.randperm(len(train.targets), generator=g)[: k["calib_transitions"]]]
    p_true = []
    with torch.no_grad():
        for b in range(0, len(pick), k["batch_size"]):
            ctx_u8, acts, _ = train.get_raw(pick[b:b + k["batch_size"]])
            probs = torch.softmax(prior(codec.decode(ctx_u8.to(device))), 1).cpu()
            p_true.append(probs[torch.arange(len(acts)), acts[:, -1]])      # acts[:, -1] = codes[i-1], the true code
    p_true = torch.cat(p_true).numpy()
    tau_legal = float(np.percentile(p_true, k["tau_legal_percentile"]))
    out = {"dir": dirs, "tau_legal": tau_legal, "tau_env": k["tau_env"], "L1_all_directions": not missing,
           "missing_directions": missing, "table": table, "calib_transitions": int(len(p_true)),
           "p_true_quantiles": {q: float(np.percentile(p_true, q)) for q in (1, 5, 25, 50)}, "prior_step": ck["step"]}
    path = ROOT / cfg["train"]["checkpoint_dir"] / "keymap.json"
    json.dump(out, open(path, "w"), indent=2)
    for row in table:
        print(f"  code {row['code']}: {row['dir']:5s} purity {row['purity']:.2f}  share {row['share']:.3f}  "
              f"still {row['still_frac']:.2f}  undetected {row['undetected_frac']:.2f}  votes {row['votes']}")
    print(f"tau_legal {tau_legal:.4g} (percentile {k['tau_legal_percentile']} of p(true code)), tau_env {k['tau_env']}; "
          f"L1 {'pass' if not missing else 'FAIL, missing ' + str(missing)}; wrote {path}")


if __name__ == "__main__":
    main()
