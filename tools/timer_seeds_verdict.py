"""Apply the pre-registered criteria of the training-seed replication (notes/timer_rule_design.md section 12) to
eval/results/timer/<layout>-N<N>-s<S>.json. Written before any replication result existed. Bars are those of the
original experiments (R1 for an on-frame cell, R2 for a gap cell; "parks" = release fraction below the R2 release bar).

  Stage 1 (4 new seeds each): claim A = S10b N=80 passes R1; claim B = S10 N=80 parks.
      REPLICATES if A and B each hold in >= min_replicate seeds; DOES NOT REPLICATE if either holds in <= max_fail;
      MIXED otherwise.
  Stage 2 (2 new seeds each): the far-gap cells park and the near-gap cells do not.
      PATTERN HOLDS if >= far_min_park of the 6 far-gap runs park and >= near_min_release of the 4 near-gap runs do
      not park; otherwise PATTERN DOES NOT HOLD.
Every seed is listed, with seed 0 (the original, already scored) alongside for reference; seed 0 never counts.

  python tools/timer_seeds_verdict.py [--config configs/timer.yaml]  -> eval/results/timer/seeds_verdict.json
"""
import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def score(r):
    m, i = r["model"]["point"], r["ideal"]["point"]
    k1 = f"on_time_tol{r['tols'][0]:g}"
    r1_bar = min(0.80, i[k1] - 0.10)
    rel_bar = min(0.90, i["release_frac"] - 0.10)
    return {"D0": r["D0_detector_pass"], "on_time": m[k1], "r1_bar": r1_bar, "passes_R1": m[k1] >= r1_bar,
            "release": m["release_frac"], "release_ci95": r["model"]["ci95"]["release_frac"], "release_bar": rel_bar,
            "parks": m["release_frac"] < rel_bar, "median_abs_err": m["median_abs_err"],
            "passes_R2": m["release_frac"] >= rel_bar and m["median_abs_err"] <= i["median_abs_err"] + 3}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/timer.yaml")
    a = p.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    rep, out_dir = cfg["seed_replication"], ROOT / cfg["eval"]["out_dir"]
    res = {"missing": [], "cells": {}}
    for cell, seeds in rep["cells"].items():
        lay, N = cell.split(":")
        runs = {}
        sha0 = None
        for s in [0] + list(seeds):
            f = out_dir / f"{lay}-N{N}{f'-s{s}' if s else ''}.json"
            r = json.load(open(f)) if f.exists() else {}
            # a result counts only if it is the fully trained model of this seed, scored on the same test set as seed 0
            ok = ("model" in r and r.get("model_step") == cfg["train"]["steps"]
                  and (s == 0 or (str(r.get("model_checkpoint", "")).endswith(f"{lay}-N{N}-s{s}/model.pt")
                                  and r.get("test_sha256") == sha0)))
            if s == 0 and ok:
                sha0 = r["test_sha256"]
            if not ok:
                res["missing"].append(f.name)
                continue
            runs[s] = score(r)
        res["cells"][cell] = runs
    new = lambda cell, key: [v[key] for s, v in res["cells"][cell].items() if s != 0]
    a_hold, b_hold = sum(new("S10b:80", "passes_R1")), sum(new("S10:80", "parks"))
    far = sum(sum(new(c, "parks")) for c in rep["far_gaps"])
    near = sum(sum(not x for x in new(c, "parks")) for c in rep["near_gaps"])
    def gate(cells):                    # each stage is scored on its own files
        n = sum(len(res["cells"][c]) for c in cells)
        want = sum(len(rep["cells"][c]) + 1 for c in cells)
        if n < want:
            return f"INCOMPLETE ({want - n} results missing)"
        if not all(v["D0"] for c in cells for v in res["cells"][c].values()):
            return "INVALID (D0 failed)"
        return None

    v1 = gate(rep["stage1"]) or ("REPLICATES" if min(a_hold, b_hold) >= rep["min_replicate"]
                                 else "DOES NOT REPLICATE" if min(a_hold, b_hold) <= rep["max_fail"] else "MIXED")
    v2 = gate(rep["far_gaps"] + rep["near_gaps"]) or (
        "PATTERN HOLDS" if far >= rep["far_min_park"] and near >= rep["near_min_release"] else "PATTERN DOES NOT HOLD")
    res.update(stage1={"verdict": v1, "S10b_N80_passes_R1": a_hold, "S10_N80_parks": b_hold},
               stage2={"verdict": v2, "far_gap_runs_parking": far, "near_gap_runs_not_parking": near})
    json.dump(res, open(out_dir / "seeds_verdict.json", "w"), indent=2)
    print(f"stage 1: {v1} (S10b N=80 passes R1 in {a_hold} new seeds; S10 N=80 parks in {b_hold})")
    print(f"stage 2: {v2} (far-gap runs parking {far}; near-gap runs not parking {near})")
    for cell, runs in res["cells"].items():
        for s, v in runs.items():
            print(f"  {cell:8s} seed {s}{' (original)' if s == 0 else ''}: on-time {v['on_time']:.3f} (R1 bar {v['r1_bar']:.3f}) "
                  f"release {v['release']:.3f} (bar {v['release_bar']:.3f}) median err {v['median_abs_err']:.1f} "
                  f"{'parks' if v['parks'] else 'releases'}")


if __name__ == "__main__":
    main()
