"""Do unexplained ghost disappearances coincide with the model's own frightened (blue) phases?

Reads residual_disappearances.csv (eval/residual_ghosts.py) and the saved frames of ctx6s16 and Model 1 and counts
frightened-blue sprite pixels in the 15 steps after each disappearance, against random frames as a control.
Thresholds (colour distance 45, background residual 40, 8 pixels) are fixed here because this is a one-off check.

  python eval/frightened_check.py
"""
import csv, sys, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "eval")
from tools.ghost_visibility import FRIGHTENED
import detectors as D
from common import load_config
cols = np.array(FRIGHTENED, float).reshape(-1, 3)
def blue(frame, bg):
    f = frame.astype(float)
    near = np.min(np.linalg.norm(f[:, :, None, :] - cols[None, None], axis=-1), axis=-1) < 45
    return int((near & (np.linalg.norm(f - bg, axis=-1) > 40)).sum())
for run, cfgp in (("m1-2M-ctx6s16", "configs/eval_m1-2M-ctx6s16.yaml"), ("model1", "configs/eval.yaml")):
    cfg = load_config(cfgp); ref = D.load_reference(cfg)
    base = cfg["out_dir"]
    preds = np.load(f"{base}/model1/preds_all.npy", mmap_mode="r")
    evs = [e for e in csv.DictReader(open(f"{base}/ghost_diag/residual_disappearances.csv")) if e["at_life_loss"] == "False"]
    R, H = preds.shape[0], 450
    rng = np.random.default_rng(0)
    ctrl = np.array([blue(np.asarray(preds[rng.integers(R), rng.integers(20, H)]), ref.bg64) for _ in range(400)])
    hit = np.array([max(blue(np.asarray(preds[int(e["rollout"]), min(int(e["step"]) - 1 + j, H - 1)]), ref.bg64) for j in (3, 6, 10, 15)) for e in evs])
    print(f"{run}: random frames with >=8 frightened-blue sprite px: {100 * np.mean(ctrl >= 8):.0f}% (median {np.median(ctrl):.0f} px) | "
          f"in the 15 steps after the {len(evs)} unexplained disappearances: {100 * np.mean(hit >= 8):.0f}% (median {np.median(hit):.0f} px)")
    for key in ("near_pacman", "returns_in_pen", "permanent", "at_edge", "near_other_ghost"):
        v = np.array([h for h, e in zip(hit, evs) if e[key] == "True"])
        share = 100 * np.mean(v >= 8) if len(v) else float("nan")
        print(f"    {key:17s} n={len(v):3d}  followed by frightened-blue sprites: {share:.0f}%")
    blue_ev = hit >= 8
    steps = np.array([int(e["absent_steps"]) for e in evs])
    print(f"    absent ghost-steps explained by the model's own frightened phases: {100 * steps[blue_ev].sum() / max(steps.sum(), 1):.0f}% "
          f"({steps[blue_ev].sum()} of {steps.sum()}); permanent among the blue ones: {sum(1 for h, e in zip(blue_ev, evs) if h and e['permanent'] == 'True')}")
