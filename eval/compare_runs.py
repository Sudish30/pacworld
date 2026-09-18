"""Side-by-side decision metrics for several evaluated checkpoints.

Reads, per results folder: <dir>/model1/summary.csv (eval_rollouts.py), <dir>/ghost_diag/pen_state_detection.csv
and <dir>/ghost_diag/pen_metrics.csv (pen_timer_analysis.py). Model 1's folders are eval/results/{model1,ghost_diag},
so its "results folder" is eval/results itself.

  python eval/compare_runs.py --seed 0 --runs model1=eval/results m1-2M-ctx4=eval/results/m1-2M-ctx4 ...
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from eval_rollouts import write_csv  # noqa: E402

HEADLINE = [("wall_iou", 450), ("pellet_iou", 450), ("pac_err", 15), ("pac_err", 150), ("pac_err", 450),
            ("responsiveness", 15), ("responsiveness", 150), ("responsiveness", 450), ("ghost_count", 150), ("ghost_count", 450)]


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--runs", nargs="+", required=True, help="name=results_dir pairs, in display order")
    p.add_argument("--out", default="eval/results/compare.csv")
    a = p.parse_args()
    runs = [r.split("=", 1) for r in a.runs]
    table = []   # rows: metric, then one column per run

    def add(label, values, fmt="{:.3f}"):
        table.append({"metric": label, **{n: (fmt.format(v) if v is not None else "-") for n, v in zip([n for n, _ in runs], values)}})

    def val(rows, **keys):
        for r in rows:
            if all(str(r.get(k)) == str(v) for k, v in keys.items()):
                return r
        return None

    print(f"{'metric':50s}" + "".join(f"{n:>16s}" for n, _ in runs))
    # 1. headline rollout metrics (mean over 30 rollouts) - regressions show up here
    summ = {n: read(ROOT / d / "model1" / "summary.csv") for n, d in runs if (ROOT / d / "model1" / "summary.csv").exists()}
    for metric, h in HEADLINE:
        vals = []
        for n, _ in runs:
            r = val(summ.get(n, []), metric=metric, horizon=h)
            vals.append(float(r["mean"]) if r else None)
        add(f"{metric} @ {h}", vals)
    gt = next(iter(summ.values()), [])
    r = val(gt, metric="ghost_count_gt", horizon=450)
    if r:
        table.append({"metric": "ghost_count_gt @ 450 (ground truth)", **{n: f"{float(r['mean']):.3f}" for n, _ in runs}})
    # 2. pen-timer decision metrics
    for n, d in runs:
        pass
    pst = {n: read(ROOT / d / "ghost_diag" / "pen_state_detection.csv") for n, d in runs if (ROOT / d / "ghost_diag" / "pen_state_detection.csv").exists()}
    pm = {n: read(ROOT / d / "ghost_diag" / "pen_metrics.csv") for n, d in runs if (ROOT / d / "ghost_diag" / "pen_metrics.csv").exists()}
    for state in ("in maze, never penned since start", "in maze, released during rollout", "in pen now"):
        for steps in ("131-250", "251-450"):
            vals = []
            for n, _ in runs:
                r = val(pst.get(n, []), condition="autoregressive, 3 steps", state=state, steps=steps)
                vals.append(float(r["model_detect"]) if r else None)
            add(f"detect | {state} @ {steps}", vals)
            r = val(next(iter(pst.values()), []), condition="autoregressive, 3 steps", state=state, steps=steps)
            if r:
                table.append({"metric": f"  ground truth", **{n: f"{float(r['gt_detect']):.3f}" for n, _ in runs}})
    for label, metric, group, b in (("pen occupancy 131-250", "pen_occupancy", "model", "131-250"),
                                     ("pen occupancy 251-450", "pen_occupancy", "model", "251-450"),
                                     ("longest pen run, median", "longest_pen_run_median", "model", ""),
                                     ("longest pen run, max", "longest_pen_run_max", "model", ""),
                                     ("rollouts with pen occupied 200+ steps", "rollouts_pen_200plus", "model", ""),
                                     ("model-world respawns", "respawns", "model", ""),
                                     ("pen runs 100+ after own respawn", "pen_runs_100plus_after_own_respawn", "model", ""),
                                     ("dwell after respawn, median", "dwell_after_respawn_median", "model (own respawns)", "")):
        vals = []
        for n, _ in runs:
            r = val(pm.get(n, []), metric=metric, group=group, bin=b)
            vals.append(float(r["value"]) if r else None)
        add(label, vals, "{:.3g}")
        gt_group = {"model": "ground truth", "model (own respawns)": "ground truth (real)"}[group]
        r = val(next(iter(pm.values()), []), metric=metric, group=gt_group, bin=b)
        if r:
            table.append({"metric": "  ground truth", **{n: f"{float(r['value']):.3g}" for n, _ in runs}})
    for b in ("1-30", "31-60", "61-90", "91-150", "151-450"):
        vals = []
        for n, _ in runs:
            r = val(pm.get(n, []), metric="release_hazard", group="model (own respawns)", bin=b)
            vals.append(float(r["value"]) if r and r["value"] not in ("nan", "") else None)
        add(f"release hazard after respawn, lag {b}", vals)
        r = val(next(iter(pm.values()), []), metric="release_hazard", group="ground truth (real)", bin=b)
        if r and r["value"] not in ("nan", ""):
            table.append({"metric": "  ground truth", **{n: f"{float(r['value']):.3f}" for n, _ in runs}})
    for row in table:
        print(f"{row['metric']:50s}" + "".join(f"{row[n]:>16s}" for n, _ in runs))
    write_csv(ROOT / a.out, table)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
