"""Stage 1 label-side evaluation of a LAM arm: the pre-registered gates G1 and G2 plus the reported agreement numbers.

EVALUATION ONLY - reads true actions and RAM of the frozen val episodes. Run it only after the arm choice has been
committed (tools/lam_checks.py --select), and only once per arm; its output decides go / stop, never a retune.

Lag 1 (fixed by the pre-registration): the code of transition i -> i+1 (codes[i]) is compared with actions[i-1],
whose effect shows in pixels one step later (eval/visible_ceiling.py).
  G1  decision events (RAM direction changes at i to a direction contained in actions[i-1]; eligible steps as in
      visible_ceiling.py): accuracy of the best many-to-one map codes[i] -> new direction, fitted on one half of the
      val episodes and scored on the other (2-fold, pooled). Pass: >= gates.g1_min and >= majority + g1_over_majority.
  G2  the keys held in actions[i-1] go through lam.resolve() with the prior on the real context up to frame i and
      c_prev = codes[i-1]; pass if dir(resolved code) == dir(codes[i]) (key-map directions, NONE for environment codes)
      at >= gates.g2_events of decision events and >= gates.g2_all of all eligible steps.
Reported, not gating: 9-way agreement with its visible-action ceiling, NMI, the confusion table, the label-optimal vs
label-free map per code, G2 restricted to direction codes (NONE == NONE removed), heading-conditioned accuracy, straight-step accuracy, ghost-turn code-switch lift, NOOP steps.

  python eval/lam_agreement.py --config configs/lam_agreement.yaml --lam-config configs/lam-A.yaml --seed 0
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from common import load_config  # noqa: E402
from dataset import FrameCodec, episode_files, gather_context, load_cache  # noqa: E402
from lam import DIRECTIONS, Prior, resolve  # noqa: E402
from metrics import ACTION_DIRS  # noqa: E402
from train_model0 import pick_device  # noqa: E402
from visible_ceiling import best_map_accuracy, eligible_steps, move_class  # noqa: E402


def nmi(a, b):
    """Normalised mutual information (arithmetic mean normalisation) of two label sequences."""
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    pa, pb = Counter(a.tolist()), Counter(b.tolist())
    pab = Counter(zip(a.tolist(), b.tolist()))
    mi = sum(c / n * np.log(c * n / (pa[x] * pb[y])) for (x, y), c in pab.items())
    h = lambda p: -sum(c / n * np.log(c / n) for c in p.values())
    denom = (h(pa) + h(pb)) / 2
    return float(mi / denom) if denom > 0 else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/lam_agreement.yaml")
    p.add_argument("--lam-config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--codes", help="use this codes array instead of the arm's (positive controls only)")
    p.add_argument("--no-g2", action="store_true", help="skip G2 (positive controls without a prior or key map)")
    p.add_argument("--out", help="default eval/results/lam/<arm>/agreement.json")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    cfg, lcfg = load_config(ROOT / a.config), load_config(ROOT / a.lam_config)
    c, gates = cfg["decision_events"], cfg["gates"]
    if c["lag"] != 1:
        raise SystemExit("decision_events.lag is fixed at 1 by the pre-registration")
    d = cfg["data"]
    d["folder"] = [f for f in ([d["folder"]] if isinstance(d["folder"], str) else d["folder"]) if (ROOT / f).exists()]
    val = set(json.load(open(ROOT / d["val_episodes"]))["val_episode_seeds"])
    files = sorted((f for f in episode_files(d, ROOT) if int(f.stem.split("_")[1]) in val), key=lambda f: int(f.stem.split("_")[1]))
    cache = load_cache(lcfg, mmap=True)
    cache.pop("actions")                                     # this script reads actions from the raw episodes only
    seed_to_ep = {int(s): e for e, s in enumerate(cache["ep_seed"])}
    codes = np.load(ROOT / (a.codes or lcfg["data"]["lam_codes"])).astype(np.int64)
    ck_dir = ROOT / lcfg["train"]["checkpoint_dir"]
    arm = Path(a.lam_config).stem
    print(f"{len(files)} of {len(val)} frozen val episodes; codes {a.codes or lcfg['data']['lam_codes']}")

    run_g2 = not a.no_g2
    if run_g2:
        keymap = json.load(open(ck_dir / "keymap.json"))
        dirs_of = keymap["dir"]
        device = pick_device(lcfg["train"]["device"])
        ck = torch.load(ck_dir / "prior.pt", map_location=device)
        prior = Prior(ck["cfg"]).to(device).eval()
        prior.load_state_dict(ck["model"])
        codec = FrameCodec(cache.get("palette")).to(device)
        frames = torch.from_numpy(np.asarray(cache["frames"])) if not isinstance(cache["frames"], np.memmap) else torch.from_numpy(cache["frames"])
        offsets = torch.as_tensor(lcfg["data"]["context_offsets"], dtype=torch.long)

    R = {k: [] for k in ("ev_code", "ev_dir", "ev_prev", "ev_fold", "all_code", "all_act", "all_fold",
                         "st_code", "st_move", "st_fold", "sw", "ghost_turn", "g2_all", "g2_ev", "g2_noop",
                         "g2_all_dironly", "g2_ev_dironly", "lam_none_all", "lam_none_ev")}
    for n, f in enumerate(files):
        e = seed_to_ep[int(f.stem.split("_")[1])]
        g0, g1_ = int(cache["ep_start"][e]), int(cache["ep_start"][e + 1])
        ep = np.load(f)
        ram, acts = ep["ram"], ep["actions"]
        T = len(acts)
        assert g1_ - g0 == T + 1, f"{f}: cache has {g1_ - g0} frames, the recording {T + 1}"
        cds = codes[g0:g1_]
        fold = n % c["folds"]
        dirs = ram[:, c["direction_ram"]].astype(int) & 3
        x, y = ram[:, c["pac_x_ram"]].astype(float), ram[:, c["pac_y_ram"]].astype(float)
        gx, gy = ram[:, c["ghost_x_ram"]].astype(float), ram[:, c["ghost_y_ram"]].astype(float)
        ok = eligible_steps(ram, c)
        idx = [i for i in range(2, T) if ok[i] and ok[i + 1]]
        ev = {i for i in idx if dirs[i] != dirs[i - 1] and dirs[i] in ACTION_DIRS[int(acts[i - 1])]}
        for i in idx:
            R["all_code"].append(int(cds[i]))
            R["all_act"].append(int(acts[i - 1]))
            R["all_fold"].append(fold)
            if i in ev:
                R["ev_code"].append(int(cds[i]))
                R["ev_dir"].append(DIRECTIONS[dirs[i]])
                R["ev_prev"].append(DIRECTIONS[dirs[i - 1]])
                R["ev_fold"].append(fold)
            elif dirs[i] == dirs[i - 1]:
                mv = move_class(x[i] - x[i - 1], y[i] - y[i - 1])
                if mv != "none":
                    R["st_code"].append(int(cds[i]))
                    R["st_move"].append(mv)
                    R["st_fold"].append(fold)
                turned = False
                for k in range(gx.shape[1]):
                    m0 = move_class(gx[i - 1, k] - gx[i - 2, k], gy[i - 1, k] - gy[i - 2, k])
                    m1 = move_class(gx[i, k] - gx[i - 1, k], gy[i, k] - gy[i - 1, k])
                    jump = max(abs(gx[i, k] - gx[i - 1, k]) + abs(gy[i, k] - gy[i - 1, k]),
                               abs(gx[i - 1, k] - gx[i - 2, k]) + abs(gy[i - 1, k] - gy[i - 2, k])) > c["max_step"]
                    turned |= (m0 != "none" and m1 != "none" and m0 != m1 and not jump)
                R["sw"].append(int(cds[i] != cds[i - 1]))
                R["ghost_turn"].append(int(turned))
        if run_g2 and idx:
            probs = []
            with torch.no_grad():
                for b in range(0, len(idx), 512):
                    tg = torch.tensor([g0 + i + 1 for i in idx[b:b + 512]])        # context ends at frame i
                    f_, _, _ = gather_context(frames, frames[:, :1, :1], tg, torch.full_like(tg, g0), offsets)
                    probs.append(torch.softmax(prior(codec.decode(f_.to(device))), 1).cpu().numpy())
            probs = np.concatenate(probs)
            for i, pr in zip(idx, probs):
                keys = {DIRECTIONS[k] for k in ACTION_DIRS[int(acts[i - 1])]}
                ch = resolve(keys, pr, int(cds[i - 1]), keymap)
                agree = int(dirs_of[ch] == dirs_of[int(cds[i])])
                none = dirs_of[int(cds[i])] == "NONE"          # NONE == NONE counts as agreement in G2 (as registered)
                R["g2_all"].append(agree)
                R["lam_none_all"].append(int(none))
                if not none:
                    R["g2_all_dironly"].append(agree)
                if i in ev:
                    R["g2_ev"].append(agree)
                    R["lam_none_ev"].append(int(none))
                    if not none:
                        R["g2_ev_dironly"].append(agree)
                if int(acts[i - 1]) == 0:
                    R["g2_noop"].append(agree)
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(files)} episodes, {len(R['ev_code'])} decision events")

    maj = Counter(R["ev_dir"]).most_common(1)[0][1] / len(R["ev_dir"])
    g1 = best_map_accuracy(R["ev_code"], R["ev_dir"], R["ev_fold"])
    conf = {}
    for code, dd in zip(R["ev_code"], R["ev_dir"]):
        conf.setdefault(code, Counter())[dd] += 1
    label_opt = {int(k): v.most_common(1)[0][0] for k, v in conf.items()}
    sw, gt = np.array(R["sw"]), np.array(R["ghost_turn"])
    lift = float(sw[gt == 1].mean() - sw[gt == 0].mean()) if gt.any() and (gt == 0).any() else float("nan")
    ceiling_file = next((f for f in (ROOT / "eval/results/lam/visible_ceiling_full.json", ROOT / "eval/results/lam/visible_ceiling.json")
                         if f.exists()), ROOT / "eval/results/lam/visible_ceiling.json")   # the full 388-episode split if measured
    ceil = json.load(open(ceiling_file))["ceiling"] if ceiling_file.exists() else {}
    res = {"arm": arm, "codes": a.codes or lcfg["data"]["lam_codes"], "episodes": len(files), "decision_events": len(R["ev_code"]),
           "eligible_steps": len(R["all_code"]), "lag": 1,
           "G1": {"accuracy": g1, "majority_rate": maj, "min": gates["g1_min"], "min_over_majority": maj + gates["g1_over_majority"],
                  "pass": g1 >= gates["g1_min"] and g1 >= maj + gates["g1_over_majority"]},
           "reported": {
               "action9_all_steps": best_map_accuracy(R["all_code"], R["all_act"], R["all_fold"]),
               "action9_all_steps_ceiling": ceil.get("all eligible steps: ram1 -> action9"),
               "nmi_code_action9_all_steps": nmi(R["all_code"], R["all_act"]),
               "heading_conditioned_G1": best_map_accuracy([f"{k}|{p}" for k, p in zip(R["ev_code"], R["ev_prev"])], R["ev_dir"], R["ev_fold"]),
               "straight_step_accuracy": best_map_accuracy(R["st_code"], R["st_move"], R["st_fold"]) if R["st_code"] else None,
               "ghost_turn_code_switch_lift": lift, "code_switch_rate_straight": float(sw.mean()) if len(sw) else None,
               "confusion_code_x_new_direction": {int(k): dict(v) for k, v in sorted(conf.items())},
               "label_optimal_map": label_opt}}
    if run_g2:
        g2e, g2a = float(np.mean(R["g2_ev"])), float(np.mean(R["g2_all"]))
        res["G2"] = {"events": g2e, "all_steps": g2a, "min_events": gates["g2_events"], "min_all": gates["g2_all"],
                     "pass": g2e >= gates["g2_events"] and g2a >= gates["g2_all"],
                     "noop_steps": float(np.mean(R["g2_noop"])) if R["g2_noop"] else None, "noop_n": len(R["g2_noop"])}
        # reported, not gating: G2 where the LAM's own code has a direction (NONE == NONE agreements removed)
        mean_or_none = lambda v: float(np.mean(v)) if v else None
        res["reported"]["G2_direction_codes_only"] = {"events": mean_or_none(R["g2_ev_dironly"]), "all_steps": mean_or_none(R["g2_all_dironly"]),
                                                      "lam_code_NONE_share_events": mean_or_none(R["lam_none_ev"]),
                                                      "lam_code_NONE_share_all": mean_or_none(R["lam_none_all"])}
        res["reported"]["label_free_map"] = {i: dd for i, dd in enumerate(dirs_of)}
        ev_counts = Counter(R["ev_code"])
        res["reported"]["label_free_vs_label_optimal_event_share_agreeing"] = float(
            sum(v for k, v in ev_counts.items() if label_opt.get(k) == dirs_of[k]) / len(R["ev_code"]))
    out = ROOT / (a.out or f"eval/results/lam/{arm}/agreement.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "reported"}, indent=2))
    print(json.dumps({k: v for k, v in res["reported"].items() if not k.startswith("confusion")}, indent=2))
    verdict = [f"G1 {'PASS' if res['G1']['pass'] else 'FAIL'} ({g1:.4f})"] + ([f"G2 {'PASS' if res['G2']['pass'] else 'FAIL'} "
               f"({res['G2']['events']:.4f} events / {res['G2']['all_steps']:.4f} all)"] if run_g2 else [])
    print(" | ".join(verdict) + f"; wrote {out}")


if __name__ == "__main__":
    main()
