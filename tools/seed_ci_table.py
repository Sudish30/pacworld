"""Appendix table of the training-seed replication: every run with its release fraction, on-time fraction and their
95% bootstrap CIs over the 20 test episodes, read from eval/results/timer/<layout>-N<N>[-s<S>].json. Prints LaTeX
rows (pasted into paper/main.tex, appendix) and writes results/supplementary_results.md.

  python tools/seed_ci_table.py --config configs/timer.yaml --seed 0
"""
import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/timer.yaml")
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    rep, out_dir = cfg["seed_replication"], ROOT / cfg["eval"]["out_dir"]
    tex, md = [], ["# Supplementary results", "", "Training-seed replication of the synthetic follow-up (pre-registration: "
                   "`notes/timer_rule_design.md` section 12). Each row is one trained model; seed 0 is the original run. "
                   "Brackets: 95% bootstrap CIs over the 20 test episodes (60 starts). A gap cell parks when its release "
                   "fraction is below its bar.", "", "| cell | seed | release [CI] | release bar | on-time, tol 2 [CI] | R1 bar | parked of 60 |",
                   "|---|---|---|---|---|---|---|"]
    for cell, seeds in rep["cells"].items():
        lay, N = cell.split(":")
        for s in [0] + list(seeds):
            r = json.load(open(out_dir / f"{lay}-N{N}{f'-s{s}' if s else ''}.json"))
            m, ci, i = r["model"]["point"], r["model"]["ci95"], r["ideal"]["point"]
            k1 = f"on_time_tol{r['tols'][0]:g}"
            rb, r1 = min(0.90, i["release_frac"] - 0.10), min(0.80, i[k1] - 0.10)
            parked = sum(x < 0 for x in r["model"]["lags"])
            rel = f"{m['release_frac']:.3f} [{ci['release_frac'][0]:.2f}, {ci['release_frac'][1]:.2f}]"
            ont = f"{m[k1]:.3f} [{ci[k1][0]:.2f}, {ci[k1][1]:.2f}]"
            tex.append(f"{lay} $N{{=}}{N}$ & {s} & {rel} & {rb:.3f} & {ont} & {r1:.3f} & {parked} \\\\")
            md.append(f"| {lay} N={N} | {s} | {rel} | {rb:.3f} | {ont} | {r1:.3f} | {parked} |")
    print("\n".join(tex))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/supplementary_results.md").write_text("\n".join(md) + "\n")
    print(f"wrote results/supplementary_results.md ({len(md) - 6} rows)")


if __name__ == "__main__":
    main()
