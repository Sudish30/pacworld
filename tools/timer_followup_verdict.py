"""Apply the pre-registered criteria of the hidden-timer FOLLOW-UP (notes/timer_rule_design.md section 10) to the
follow-up cells in eval/results/timer/. Written before any follow-up model result existed. Scoring as in part a:

  on-frame cell (S10b N=80)       R1: model on-time (tol 2) >= min(0.80, ideal - 0.10)
  off-frame cells (S10b 72/79/88) R2: model release >= min(0.90, ideal release - 0.10) and median |lag-N| <= ideal median + 3
                                  "parks" = the release part of R2 fails
  D0 every cell                   the detector gives lag = N on every real start
Owner's prediction: the on-frame cell passes R1 and every off-frame cell parks.
Verdict: SUPPORTED (on-frame passes, all off-frame park) / PARTLY SUPPORTED (on-frame passes, some off-frame park) /
ON-FRAME ONLY (on-frame passes, no off-frame parks) / NOT SUPPORTED (on-frame fails R1).

  python tools/timer_followup_verdict.py [--config configs/timer.yaml]  -> eval/results/timer/followup_verdict.json
"""
import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ON_FRAME = {"S10b": [80]}
OFF_FRAME = {"S10b": [72, 79, 88]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/timer.yaml")
    a = p.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    out_dir = ROOT / cfg["eval"]["out_dir"]
    res = {"missing": [], "cells": {}}
    for lay, Ns in cfg["followup_grid"].items():
        for N in Ns:
            f = out_dir / f"{lay}-N{N}.json"
            if not f.exists() or "model" not in json.load(open(f)):
                res["missing"].append(f"{lay}-N{N}")
                continue
            r = json.load(open(f))
            m, i = r["model"]["point"], r["ideal"]["point"]
            k1 = f"on_time_tol{r['tols'][0]:g}"
            c = {"D0": r["D0_detector_pass"], "model": m, "model_ci95": r["model"]["ci95"], "ideal": i}
            if N in ON_FRAME.get(lay, []):
                bar = min(0.80, i[k1] - 0.10)
                c["R1"] = {"value": m[k1], "bar": bar, "pass": m[k1] >= bar}
            if N in OFF_FRAME.get(lay, []):
                rb, mb = min(0.90, i["release_frac"] - 0.10), i["median_abs_err"] + 3
                c["R2"] = {"release": m["release_frac"], "release_bar": rb, "median_err": m["median_abs_err"], "median_bar": mb,
                           "pass": m["release_frac"] >= rb and m["median_abs_err"] <= mb, "parks": m["release_frac"] < rb}
            res["cells"][f"{lay}-N{N}"] = c
    cells = res["cells"].values()
    on = [c["R1"]["pass"] for c in cells if "R1" in c]
    parks = [c["R2"]["parks"] for c in cells if "R2" in c]
    if res["missing"]:
        verdict = f"INCOMPLETE ({len(res['missing'])} cells missing)"
    elif not all(c["D0"] for c in cells):
        verdict = "INVALID (D0 failed)"
    elif not all(on):
        verdict = "NOT SUPPORTED"
    elif all(parks):
        verdict = "SUPPORTED"
    elif any(parks):
        verdict = "PARTLY SUPPORTED"
    else:
        verdict = "ON-FRAME ONLY"
    res["verdict"] = verdict
    json.dump(res, open(out_dir / "followup_verdict.json", "w"), indent=2)
    print(f"follow-up verdict: {verdict}")
    for name, c in res["cells"].items():
        tag = f"R1 {'pass' if c['R1']['pass'] else 'FAIL'}" if "R1" in c else f"R2 {'pass' if c['R2']['pass'] else 'FAIL'}{' (parks)' if c['R2']['parks'] else ''}"
        print(f"  {name:10s} {tag:16s} model {json.dumps({k: round(v, 3) for k, v in c['model'].items()})}")


if __name__ == "__main__":
    main()
