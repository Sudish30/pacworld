"""Visible-action ceiling: how much of the true action can ANY frame-based code recover? No model involved.

EVALUATION ONLY - reads true actions and RAM of the frozen val episodes.

At decision events (Pac-Man's RAM direction changes at frame i to a direction contained in actions[i-1], the
metrics.responsiveness_events rule; life-loss windows, the episode start and tunnel wraps excluded) the "code" is
replaced by Pac-Man's real movement over the transition i-1 -> i, the only transition a latent action for step i-1
can see. The best many-to-one mapping from that movement to a label is fitted on one half of the episodes and scored
on the other (2-fold, pooled). This is the most a code that perfectly tracks Pac-Man's motion could score.

Movement features: RAM one-step (i-1 -> i), RAM two-step (i-1 -> i+1, reference only), and the 64px pixel detector
over i-1 -> i ("pix1") and over i -> i+1 ("pix_lag1"). The frames lag RAM by about one step, so the turn caused by
actions[i-1] shows in pixels mainly over i -> i+1; the latent-action gates therefore compare codes[i] with
actions[i-1] (lag 1, fixed by the pre-registration), and "pix_lag1 -> new_direction" is the pre-registered
pixel ceiling (the full-split value must be >= 0.76 before Stage 1, else stop and ask). Labels: the new RAM direction (G1's label) and the
true 9-way ALE action. The same is reported over all eligible steps (the ceiling for 9-way all-step agreement).

  python eval/visible_ceiling.py --config configs/lam_agreement.yaml --seed 0
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from common import load_config  # noqa: E402
from dataset import episode_files  # noqa: E402
import detectors as D  # noqa: E402
from metrics import ACTION_DIRS  # noqa: E402

DIR_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]


def eligible_steps(ram, c):
    """Boolean over frames i: the transition i-1 -> i is outside the episode start, life-loss windows and wraps."""
    T = len(ram)
    ok = np.ones(T, bool)
    ok[: c["start_of_life_steps"] + 1] = False
    lives = ram[:, c["lives_ram"]].astype(int)
    for t in np.nonzero(np.diff(lives) < 0)[0]:
        ok[max(0, t - c["death_anim_steps"]): t + c["start_of_life_steps"]] = False
    x, y = ram[:, c["pac_x_ram"]].astype(int), ram[:, c["pac_y_ram"]].astype(int)
    jump = np.zeros(T, bool)
    jump[1:] = (np.abs(np.diff(x)) + np.abs(np.diff(y))) > c["max_step"]
    return ok & ~jump


def move_class(dx, dy, thresh=0.0):
    """Dominant-axis movement class; 'none' below the threshold. Orientation does not matter (the mapping is learnt)."""
    if abs(dx) <= thresh and abs(dy) <= thresh:
        return "none"
    return ("x+" if dx > 0 else "x-") if abs(dx) >= abs(dy) else ("y+" if dy > 0 else "y-")


def best_map_accuracy(feat, label, fold):
    """2-fold accuracy of the majority-vote mapping feature -> label (unseen features get the fold's majority label)."""
    feat, label, fold = np.asarray(feat, object), np.asarray(label, object), np.asarray(fold)
    hits = 0
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        by = {}
        for f, l in zip(feat[tr], label[tr]):
            by.setdefault(f, Counter())[l] += 1
        default = Counter(label[tr]).most_common(1)[0][0]
        m = {f: c.most_common(1)[0][0] for f, c in by.items()}
        hits += sum(m.get(f, default) == l for f, l in zip(feat[te], label[te]))
    return hits / len(label)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/lam_agreement.yaml")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--no-detector", action="store_true", help="skip the pixel-detector feature (faster)")
    p.add_argument("--out", default="eval/results/lam/visible_ceiling.json")
    a = p.parse_args()
    np.random.seed(a.seed)
    cfg = load_config(ROOT / a.config)
    c = cfg["decision_events"]
    d = cfg["data"]
    d["folder"] = [f for f in ([d["folder"]] if isinstance(d["folder"], str) else d["folder"]) if (ROOT / f).exists()]
    val = set(json.load(open(ROOT / d["val_episodes"]))["val_episode_seeds"])
    files = [f for f in episode_files(d, ROOT) if int(f.stem.split("_")[1]) in val]
    print(f"{len(files)} of {len(val)} frozen val episodes found in {d['folder']}")
    ref = None if a.no_detector else D.load_reference(cfg)

    ev = {k: [] for k in ("ram1", "ram2", "pix1", "pix_lag1", "dir", "act", "fold", "prev")}
    allsteps = {k: [] for k in ("ram1", "act", "fold")}
    for n, f in enumerate(sorted(files, key=lambda f: int(f.stem.split("_")[1]))):
        e = np.load(f)
        ram, acts = e["ram"], e["actions"]
        fold = n % c["folds"]
        dirs = ram[:, c["direction_ram"]].astype(int) & 3
        x, y = ram[:, c["pac_x_ram"]].astype(float), ram[:, c["pac_y_ram"]].astype(float)
        ok = eligible_steps(ram, c)
        T = len(acts)                                   # frames 0..T; transitions i-1 -> i for i in 1..T
        idx = [i for i in range(1, T) if ok[i] and ok[i + 1]]
        for i in idx:
            allsteps["ram1"].append(move_class(x[i] - x[i - 1], y[i] - y[i - 1]))
            allsteps["act"].append(int(acts[i - 1]))
            allsteps["fold"].append(fold)
        events = [i for i in idx if dirs[i] != dirs[i - 1] and dirs[i] in ACTION_DIRS[int(acts[i - 1])]]
        if events and ref is not None:
            need = sorted({j for i in events for j in (i - 1, i, i + 1)})
            small = D.downsample_frames(e["frames"][need], ref.size, ref.mode)
            pos = {j: ref.sprites(fr)["pac"] for j, fr in zip(need, small)}
        for i in events:
            ev["ram1"].append(move_class(x[i] - x[i - 1], y[i] - y[i - 1]))
            ev["ram2"].append(move_class(x[i + 1] - x[i - 1], y[i + 1] - y[i - 1]))
            if ref is not None:
                p0, p1 = pos[i - 1], pos[i]
                ev["pix1"].append("undetected" if p0 is None or p1 is None else
                                  move_class(p1[1] - p0[1], p1[0] - p0[0], c["min_move_px"]))
                p2 = pos[i + 1]                          # lag 1: the transition i -> i+1, where the turn shows in pixels
                ev["pix_lag1"].append("undetected" if p1 is None or p2 is None else
                                      move_class(p2[1] - p1[1], p2[0] - p1[0], c["min_move_px"]))
            ev["dir"].append(DIR_NAMES[dirs[i]])
            ev["act"].append(int(acts[i - 1]))
            ev["prev"].append(DIR_NAMES[dirs[i - 1]])
            ev["fold"].append(fold)
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(files)} episodes, {len(ev['dir'])} decision events")

    fold = ev["fold"]
    maj = lambda lab: Counter(lab).most_common(1)[0][1] / len(lab)
    res = {"episodes": len(files), "decision_events": len(ev["dir"]), "eligible_steps": len(allsteps["act"]),
           "majority_rate": {"new_direction": maj(ev["dir"]), "action9": maj(ev["act"])},
           "direction_counts": dict(Counter(ev["dir"])), "action_counts": {str(k): v for k, v in sorted(Counter(ev["act"]).items())},
           "ram1_none_frac": ev["ram1"].count("none") / len(ev["ram1"]), "ceiling": {}}
    feats = ["ram1", "ram2"] + ([] if ref is None else ["pix1", "pix_lag1"])
    for ft in feats:
        res["ceiling"][f"{ft} -> new_direction"] = best_map_accuracy(ev[ft], ev["dir"], fold)
        res["ceiling"][f"{ft} -> action9"] = best_map_accuracy(ev[ft], ev["act"], fold)
    res["ceiling"]["ram1+prev_direction -> new_direction"] = best_map_accuracy([f"{m}|{p}" for m, p in zip(ev["ram1"], ev["prev"])], ev["dir"], fold)
    res["ceiling"]["all eligible steps: ram1 -> action9"] = best_map_accuracy(allsteps["ram1"], allsteps["act"], allsteps["fold"])
    res["majority_rate"]["all steps action9"] = maj(allsteps["act"])
    if ref is not None:
        res["pix1_class_counts"] = dict(Counter(ev["pix1"]))
    # binomial standard error of each ceiling (events are not fully independent; this is a lower bound)
    res["n_for_se"] = len(ev["dir"])
    print(json.dumps(res, indent=2))
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
