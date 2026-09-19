"""Teacher-forced frightened-phase end: given REAL context, does the model turn the ghosts back on time?

Reads the teacher-forced detections in each run's ghost_diag/raw.npz and the RAM frightened timer. A lag of +1 step
means the model never predicts the end of a phase: it only copies it once a real coloured frame is in its context.

  python eval/tf_phase_end.py [eval config ...]
"""
import sys, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "eval")
from common import load_config
from dataset import episode_path
from pathlib import Path
CONFIGS = sys.argv[1:] or ["configs/eval_m1-2M-ctx6s16.yaml", "configs/eval_m1-2M-ctx-r148.yaml", "configs/eval.yaml"]
for run, cfgp in ((Path(c).stem[5:] if Path(c).stem.startswith("eval_") else "model1", c) for c in CONFIGS):
    cfg = load_config(cfgp)
    z = np.load(Path(cfg["out_dir"]) / "ghost_diag" / "raw.npz", allow_pickle=True)
    names = list(z["conditions"]); ci = names.index("teacher-forced, 3 steps")
    det, gt = z[f"c{ci}_det"], z["gt_det"]
    start, H = int(z["start"]), int(z["horizon"])
    rows = []
    for r, s in enumerate(z["episode_seed"]):
        ram = np.load(episode_path(cfg["data"], int(s), Path(".")))["ram"][start:start + H]
        fr = ram[:, cfg["gating"]["frightened_ram"]] > 0
        ends = [t for t in range(1, H - 12) if fr[t - 1] and not fr[t]]
        for t in ends:
            # first step at which ground truth shows >= 3 coloured ghosts again, and the same for the model
            g_first = next((k for k in range(-5, 12) if gt[r, t + k].sum() >= 3), None)
            m_first = next((k for k in range(-5, 12) if det[r, t + k].sum() >= 3), None)
            rows.append((g_first, m_first, int(gt[r, t + 3].sum()), int(det[r, t + 3].sum()), int(det[r, t - 6].sum())))
    lag = [m - g for g, m, *_ in rows if g is not None and m is not None]
    print(f"{run:16s} phase ends x seeds: {len(rows)} | coloured ghosts 3 steps after the real end: ground truth {np.mean([x[2] for x in rows]):.2f}, "
          f"model (teacher-forced) {np.mean([x[3] for x in rows]):.2f} | coloured 6 steps BEFORE the end (premature flips): {np.mean([x[4] for x in rows]):.2f} | "
          f"model flips back {np.mean(lag) if lag else float('nan'):+.1f} steps vs ground truth (n={len(lag)}, never within 12 steps: {sum(1 for g, m, *_ in rows if g is not None and m is None)})")
