"""External baseline: DIAMOND's released Atari-100k Ms. Pac-Man world model in our rollout harness, no retraining.
Pre-registered in notes/handoff.md (validity gates V-D1..V-D3, prediction P-D1). CPU is enough (4.4M params).

1. Re-simulate the held-out episodes from their seeds and recorded actions. Every step yields our frame (maze crop,
   max over 4 skipped frames), checked bit for bit against the recording (V-D1), and DIAMOND's frame (full screen,
   max over the last 2 skipped frames, exact area resize to 64x64 - cv2 INTER_AREA).
2. Roll DIAMOND out from step 100 with the recorded actions: 4-frame / 4-action conditioning, its own 3-step Euler
   sampler, 10 episodes x 3 sampler seeds, 900 steps.
3. Score with detectors rebuilt for DIAMOND's geometry (the same maze reference through DIAMOND's resize, detection
   restricted to the maze rows). Pac-Man positions are converted to our 64px-equivalent coordinates, so Pac-Man error
   and responsiveness are comparable. Pen occupancy uses the pen box mapped to DIAMOND's rows, smoothed over the
   flicker cycle. Writes per_step.csv in eval_rollouts' format (for eval/bootstrap_ci.py), pen.json, gates.json.

  python eval/diamond_baseline.py --config configs/eval_diamond.yaml --seed 0
"""
import argparse
import contextlib
import csv
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from common import crop, get_ram, load_config, make_env  # noqa: E402
from dataset import episode_files  # noqa: E402
import detectors as D  # noqa: E402
import metrics as M  # noqa: E402

OUR_NATIVE_H = 172


# ----------------------------------------------------------------------------- frames
def area_resize(native, size, native_h):
    """Exact area-average resize of (T, native_h, 160, 3) uint8 -> (T, size, size, 3) uint8 (= cv2 INTER_AREA)."""
    Mh, Mw = D._area_matrix(native_h, size), D._area_matrix(native.shape[2], size)
    f = native.astype(np.float64)
    out = np.tensordot(np.tensordot(f, Mh, axes=([1], [1])), Mw, axes=([1], [1]))   # (T, C, a, b)
    out = out.transpose(0, 2, 3, 1)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def resimulate(ep, rcfg, pool_last):
    """Replay a recorded episode; returns (our frames, DIAMOND native frames (T, 210, 160, 3)) and asserts V-D1."""
    env = make_env(rcfg)
    obs, _ = env.reset(seed=int(ep["seed"]))
    ours, dia = [crop(obs, rcfg)], [obs]
    for a in ep["actions"]:
        raw = []
        for _ in range(rcfg["frame_skip"]):
            frame, _, term, trunc, _ = env.step(int(a))
            raw.append(frame)
            if term or trunc:
                break
        ours.append(crop(np.max(np.stack(raw[-rcfg["max_pool_last"]:]), 0), rcfg))
        dia.append(np.max(np.stack(raw[-pool_last:]), 0))
    env.close()
    ours = np.stack(ours)
    return ours, np.stack(dia), bool(np.array_equal(ours, ep["frames"]))


# ----------------------------------------------------------------------------- detectors in DIAMOND's geometry
class DiamondReference(D.MazeReference):
    """The same maze reference, built from a full-screen native frame through DIAMOND's resize; detection is limited
    to the rows that lie inside the maze (the score and lives area below it changes during play)."""

    def __init__(self, native_full, cfg, size, native_h, maze_rows):
        patch = {"NATIVE_H": native_h, "downsample_frames": lambda fr, s=size, m="box": area_resize(fr, s, native_h)}
        saved = {k: getattr(D, k) for k in patch}
        try:
            for k, v in patch.items():
                setattr(D, k, v)
            super().__init__(native_full, cfg, size, "box")
        finally:
            for k, v in saved.items():
                setattr(D, k, v)
        self.maze_rows = maze_rows
        n_before = len(self.pellets)
        self.pellets = [q for q in self.pellets if q["cells"][:, 0].max() < maze_rows]   # score digits are not pellets
        self.dropped_pellets = n_before - len(self.pellets)
        self.pellet_cells = np.zeros((size, size), bool)
        for q in self.pellets:
            self.pellet_cells[q["cells"][:, 0], q["cells"][:, 1]] = True
        self.wall_eval = ~self.pellet_cells
        self.wall_eval[maze_rows:] = False
        self.wall_ref[maze_rows:] = False

    def unmix(self, frame64, presence=None):
        accept, cls, alpha, mag = super().unmix(frame64, presence)
        accept[self.maze_rows:] = False
        return accept, cls, alpha, mag


def to_ours(pos, native_h):
    """(row, col, ...) in DIAMOND's 64px frame -> (row, col) in our 64px-equivalent coordinates (cols share a scale)."""
    if pos is None:
        return None
    r = ((pos[0] + 0.5) * native_h / 64 - 0.5 + 0.5) * 64 / OUR_NATIVE_H - 0.5
    return (r, pos[1])


# ----------------------------------------------------------------------------- DIAMOND model
def load_diamond(dc, device):
    sys.modules.setdefault("omegaconf", types.SimpleNamespace(OmegaConf=None, DictConfig=dict))   # config-only import
    sys.path.insert(0, str(ROOT / dc["source"]))
    from models.diffusion import Denoiser, DenoiserConfig, DiffusionSampler, DiffusionSamplerConfig
    from models.diffusion.inner_model import InnerModelConfig
    from utils import extract_state_dict
    import hashlib
    h = hashlib.sha256(open(ROOT / dc["checkpoint"], "rb").read()).hexdigest()
    if not h.startswith(dc["checkpoint_sha256_prefix"]):
        raise SystemExit(f"DIAMOND checkpoint sha256 {h[:16]} != {dc['checkpoint_sha256_prefix']}")
    den = Denoiser(DenoiserConfig(inner_model=InnerModelConfig(**dc["inner_model"], num_actions=dc["num_actions"]),
                                  sigma_data=dc["sigma_data"], sigma_offset_noise=dc["sigma_offset_noise"]))
    den.load_state_dict(extract_state_dict(torch.load(ROOT / dc["checkpoint"], map_location="cpu"), "denoiser"))
    den.to(device).eval()
    return DiffusionSampler(den, DiffusionSamplerConfig(**dc["sampler"])), h


# ----------------------------------------------------------------------------- main
def longest(b):
    best = cur = 0
    for x in b:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/eval_diamond.yaml")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n-episodes", type=int, help="override (smoke tests)")
    p.add_argument("--horizon", type=int, help="override (smoke tests)")
    p.add_argument("--gates-only", action="store_true", help="V-D1..V-D3 on ground truth only, no DIAMOND rollouts")
    a = p.parse_args()
    cfg = load_config(ROOT / a.config)
    dc, r, pc = cfg["diamond"], cfg["rollout"], cfg["pen"]
    for k in ("n_episodes", "horizon"):
        if getattr(a, k) is not None:
            r[k] = getattr(a, k)
    rcfg = load_config(ROOT / cfg["record_config"])
    out = ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # the same episodes as eval_rollouts.load_episodes: the N longest frozen val episodes
    val = set(json.load(open(ROOT / cfg["data"]["val_episodes"]))["val_episode_seeds"])
    eps = [dict(np.load(f)) for f in episode_files(cfg["data"], ROOT) if int(f.stem.split("_")[1]) in val]
    eps.sort(key=lambda e: -len(e["actions"]))
    eps = eps[: r["n_episodes"]]
    start, H, K = r["start_step"], r["horizon"], cfg["data"]["context"]
    gates = {"V-D1_resimulation_bit_exact": True}
    for e in eps:
        _, dia_native, same = resimulate(e, rcfg, dc["pool_last"])
        gates["V-D1_resimulation_bit_exact"] &= same
        e["dia"] = area_resize(dia_native, 64, dc["native_h"])
        e["native0"] = dia_native[0]
        print(f"  episode {int(e['seed'])}: {len(e['actions'])} steps re-simulated, bit-exact {same} ({time.time() - t0:.0f}s)")
    ref = DiamondReference(eps[0]["native0"], cfg, 64, dc["native_h"], dc["maze_rows"])
    ours_ref = D.load_reference({**cfg, "data": {**cfg["data"], "resample": "box"}})
    gates["pellets_in_reference"] = [len(ref.pellets), ref.dropped_pellets, len(ours_ref.pellets)]

    def pen_occ(frames):
        box = (slice(*pc["box_rows"]), slice(*pc["box_cols"]))
        raw = np.array([(np.linalg.norm(f[box].astype(float) - ref.bg64[box], axis=-1) > pc["occ_resid"]).sum() >= pc["occ_pixels"] for f in frames])
        return np.array([raw[max(0, t - pc["smooth"] + 1):t + 1].any() for t in range(len(raw))])

    # ground truth in DIAMOND's geometry
    gts = []
    for e in eps:
        T = len(e["dia"])
        pres = [ref.pellet_presence(f) for f in e["dia"]]
        det = [ref.sprites(f, pp) for f, pp in zip(e["dia"], pres)]
        gts.append({"T": T, "pres": pres, "det": det, "wall": [ref.wall_iou(f) for f in e["dia"]],
                    "eligible": M.ghost_eligibility(e["ram"], cfg), "dir": M.gt_directions(e["ram"], cfg)})
    gated = [t for e, g in zip(eps, gts) for t in range(start, min(g["T"], start + H)) if g["eligible"][t]]
    pac_found = np.mean([gts[i]["det"][t]["pac"] is not None for i, g in enumerate(gts) for t in range(start, min(g["T"], start + H)) if g["eligible"][t]])
    gates["V-D2_gt_pac_detected"] = float(pac_found)
    gt_long = [longest(pen_occ(e["dia"][start:start + H])) for e in eps for _ in range(r["seeds"])]
    gates["V-D3_gt_longest_pen_median"] = float(np.median(gt_long))
    gates["V-D3_gt_parked_200"] = int(sum(x >= pc["parked_len"] for x in gt_long))
    gates["V-D2_pass"] = bool(pac_found >= 0.95)
    gates["V-D3_pass"] = bool(70 <= gates["V-D3_gt_longest_pen_median"] <= 95 and gates["V-D3_gt_parked_200"] == 0)
    print(f"gates on ground truth: {gates}")
    json.dump(gates, open(out / "gates.json", "w"), indent=2)
    if a.gates_only:
        return

    # DIAMOND rollouts: one batch per sampler seed (the global RNG drives DIAMOND's sampler)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler, sha = load_diamond(dc, device)
    to_t = lambda u8: torch.from_numpy(np.ascontiguousarray(u8)).permute(0, 3, 1, 2).float().div(127.5).sub(1)
    preds = np.zeros((len(eps) * r["seeds"], H, 64, 64, 3), np.uint8)
    for s in range(r["seeds"]):
        torch.manual_seed(a.seed * 1000 + s)
        obs = torch.stack([to_t(e["dia"][start - K:start]) for e in eps]).to(device)          # (B, 4, 3, 64, 64)
        acts = torch.stack([torch.from_numpy(e["actions"][start - K:start].astype(np.int64)) for e in eps]).to(device)
        for k in range(H):
            with torch.no_grad():
                x, _ = sampler.sample(obs, acts)
            u8 = x.clamp(-1, 1).add(1).mul(127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            preds[s * len(eps):(s + 1) * len(eps), k] = u8
            nxt = torch.tensor([int(e["actions"][(start + k) % len(e["actions"])]) for e in eps], device=device)
            obs = torch.cat([obs[:, 1:], x.clamp(-1, 1)[:, None]], 1)
            acts = torch.cat([acts[:, 1:], nxt[:, None]], 1)
            if (k + 1) % 150 == 0:
                print(f"  seed {s}: step {k + 1}/{H} ({time.time() - t0:.0f}s)")
    np.save(out / "preds_all.npy", preds)

    # score (per_step.csv in eval_rollouts' format; positions in our 64px-equivalent coordinates)
    rows, pen_rows = [], []
    for j in range(len(preds)):
        i, s = j % len(eps), j // len(eps)
        e, g = eps[i], gts[i]
        positions = [to_ours(g["det"][start - K + q]["pac"], dc["native_h"]) for q in range(K)]
        per = []
        for k in range(H):
            t = start + k
            m = preds[j, k]
            pm = ref.pellet_presence(m)
            dm = ref.sprites(m, pm)
            pos = to_ours(dm["pac"], dc["native_h"])
            positions.append(pos)
            row = {"model": "diamond", "episode": int(e["seed"]), "seed": s, "step": k + 1, "gt": t < g["T"],
                   "ghost_count_ungated": M.ghost_count(dm), "wall_iou": ref.wall_iou(m), "pac_detected": pos is not None,
                   "pellets_remaining": int(pm.sum()), "pac_err": np.nan, "ghost_count": np.nan, "ghost_count_gt": np.nan,
                   "ghost_abs_err": np.nan, "pellet_iou": np.nan, "wall_iou_gt": np.nan, "pellets_remaining_gt": np.nan,
                   "resp_event": 0, "resp_hit": 0}
            if t < g["T"]:
                gp = to_ours(g["det"][t]["pac"], dc["native_h"])
                if pos is not None and gp is not None:
                    row["pac_err"] = float(np.hypot(pos[0] - gp[0], pos[1] - gp[1]))
                row["wall_iou_gt"] = g["wall"][t]
                row["pellets_remaining_gt"] = int(g["pres"][t].sum())
                row["pellet_iou"] = M.pellet_iou(pm, g["pres"][t])
                if g["eligible"][t]:
                    row["ghost_count"], row["ghost_count_gt"] = M.ghost_count(dm), M.ghost_count(g["det"][t])
                    row["ghost_abs_err"] = abs(row["ghost_count"] - row["ghost_count_gt"])
            per.append(row)
        events = M.responsiveness_events(g["dir"], e["actions"], start, min(g["T"], start + H))
        for (t, _), hit in zip(events, M.responsiveness_hits(events, positions, start, K, cfg)):
            per[t - start]["resp_event"], per[t - start]["resp_hit"] = 1, int(hit)
        rows += per
        # pen occupancy and own respawns in the model world
        occ = pen_occ(preds[j])
        spawn = np.array(pc["spawn_px"])
        resp, missing, last = 0, 0, None
        for p_ in positions[K:]:
            if p_ is None:
                missing += 1
                continue
            near = np.hypot(p_[0] - spawn[0], p_[1] - spawn[1]) <= pc["spawn_r"]
            jumped = last is not None and np.hypot(p_[0] - last[0], p_[1] - last[1]) > pc["jump_px"]
            if near and (missing >= pc["respawn_min_missing"] or jumped):
                resp += 1
            missing, last = 0, p_
        pen_rows.append({"episode": int(e["seed"]), "seed": s, "longest_pen_run": longest(occ), "own_respawns": resp})
    with open(out / "per_step.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ml = [x["longest_pen_run"] for x in pen_rows]
    resp_total = sum(x["own_respawns"] for x in pen_rows)
    pen = {"per_rollout": pen_rows, "model_longest_pen_median": float(np.median(ml)), "model_parked_200": int(sum(x >= pc["parked_len"] for x in ml)),
           "own_respawns_total": resp_total, "checkpoint_sha256": sha,
           "P-D1": ("untestable (fewer than %d own respawns)" % pc["min_respawns_for_pd1"]) if resp_total < pc["min_respawns_for_pd1"]
           else ("pass" if sum(x >= pc["parked_len"] for x in ml) >= 10 else "FAIL"),
           "pen_metric_valid": gates["V-D3_pass"]}
    json.dump(pen, open(out / "pen.json", "w"), indent=2)
    print(f"pen: parked 200+ {pen['model_parked_200']}/{len(ml)}, longest median {pen['model_longest_pen_median']}, own respawns {resp_total}; "
          f"P-D1 {pen['P-D1']} (pen metric valid: {gates['V-D3_pass']}); {time.time() - t0:.0f}s; wrote {out}")


if __name__ == "__main__":
    main()
