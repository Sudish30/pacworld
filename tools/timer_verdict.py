"""Apply the pre-registered criteria of the hidden-timer experiment (notes/timer_rule_design.md section 6, with
Amendment 1) to the scored cells in eval/results/timer/. Written before any model result existed.

  R1 on-offset in reach  model on-time (tol 2) >= min(0.80, ideal - 0.10)
  R2 gaps                model release >= min(0.90, ideal release - 0.10) and median |lag-N| <= ideal median + 3
  R3 out of reach        model on-time (tol max(2, 0.1N)) <= ideal + 0.15
  R4 N in {33, 65, 97}   S10 passes R1 and C10 passes R3
  D0 every cell          the detector gives lag = N on every real start
Verdict: HOLDS (D0 and all of R1-R4) / PARTIAL (every R3 passes, some R1/R2 fail) / FAILS (any R3 fails).

  python tools/timer_verdict.py [--config configs/timer.yaml]  -> eval/results/timer/verdict.json + a markdown table
"""
import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
R1 = {"C4": [3], "C10": [3, 8, 10], "S10": [3, 17, 33, 65, 97]}
R2 = {"S10": [8, 24, 80]}
R3 = {"C4": [8, 17, 33, 65], "C10": [17, 33, 65, 97, 200], "S10": [200]}
R4 = [33, 65, 97]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/timer.yaml")
    a = p.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    out_dir = ROOT / cfg["eval"]["out_dir"]
    cells, rows, missing = {}, [], []
    for lay, Ns in cfg["grid"].items():
        for N in Ns:
            f = out_dir / f"{lay}-N{N}.json"
            if not f.exists() or "model" not in json.load(open(f)):
                missing.append(f"{lay}-N{N}")
                continue
            cells[(lay, N)] = json.load(open(f))
    res = {"missing": missing, "cells": {}}

    def pt(r, who, key):
        return r[who]["point"][key]

    for (lay, N), r in cells.items():
        t1, t2 = r["tols"]
        k1, k2 = f"on_time_tol{t1:g}", f"on_time_tol{t2:g}"
        c = {"D0": r["D0_detector_pass"], "model": r["model"]["point"], "model_ci95": r["model"]["ci95"],
             "ideal": r["ideal"]["point"]}
        if N in R1.get(lay, []):
            bar = min(0.80, pt(r, "ideal", k1) - 0.10)
            c["R1"] = {"value": pt(r, "model", k1), "bar": bar, "pass": pt(r, "model", k1) >= bar}
        if N in R2.get(lay, []):
            rb = min(0.90, pt(r, "ideal", "release_frac") - 0.10)
            mb = pt(r, "ideal", "median_abs_err") + 3
            ok = pt(r, "model", "release_frac") >= rb and pt(r, "model", "median_abs_err") <= mb
            c["R2"] = {"release": pt(r, "model", "release_frac"), "release_bar": rb,
                       "median_err": pt(r, "model", "median_abs_err"), "median_bar": mb, "pass": ok}
        if N in R3.get(lay, []):
            bar = pt(r, "ideal", k2) + 0.15
            c["R3"] = {"value": pt(r, "model", k2), "bar": bar, "pass": pt(r, "model", k2) <= bar}
        res["cells"][f"{lay}-N{N}"] = c
    r4 = {}
    for N in R4:
        s, cc = res["cells"].get(f"S10-N{N}"), res["cells"].get(f"C10-N{N}")
        r4[N] = None if s is None or cc is None else bool(s["R1"]["pass"] and cc["R3"]["pass"])
    res["R4"] = r4
    allc = res["cells"].values()
    d0 = all(c["D0"] for c in allc)
    r3_ok = all(c["R3"]["pass"] for c in allc if "R3" in c)
    r12_ok = all(c[k]["pass"] for c in allc for k in ("R1", "R2") if k in c)
    r4_ok = all(v for v in r4.values())
    if missing:
        verdict = f"INCOMPLETE ({len(missing)} cells missing)"
    elif not d0:
        verdict = "INVALID (D0 failed)"
    elif not r3_ok:
        verdict = "FAILS"
    elif r12_ok and r4_ok:
        verdict = "HOLDS"
    else:
        verdict = "PARTIAL"
    res["verdict"] = verdict
    json.dump(res, open(out_dir / "verdict.json", "w"), indent=2)
    print(f"verdict: {verdict}")
    for name, c in res["cells"].items():
        tag = " ".join(f"{k} {'pass' if c[k]['pass'] else 'FAIL'}" for k in ("R1", "R2", "R3") if k in c)
        print(f"  {name:9s} {tag:10s} model {json.dumps({k: round(v, 3) for k, v in c['model'].items()})}")
    print(f"  R4 {r4}; missing {missing}")


if __name__ == "__main__":
    main()
