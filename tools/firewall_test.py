"""Label-firewall test: run a training command twice - once as is, once with the recorded actions replaced by random
integers before anything reads them - and require bit-identical results (every logged loss, the final weights'
sha256 and, for the LAM, the final val codes). If any label reached the run, the two would differ.

  python tools/firewall_test.py --seed 0 -- python train_lam.py --config configs/lam-B-mac.yaml --seed 0 --part lam \
      --steps 20 --batch-size 8 --eval-every 10 --wandb-mode disabled

The command must accept --dump and --firewall-scramble-seed (train_lam.py and train_model1.py do). Exit code 0 = pass.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, required=True, help="seed of the random integers that replace the actions")
    p.add_argument("--report", help="append a JSON line with the verdict to this file")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    a = p.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        raise SystemExit("give the training command after --")
    tmp = Path(tempfile.mkdtemp(prefix="firewall_"))
    dumps = {}
    for name, extra in (("clean", []), ("scrambled", ["--firewall-scramble-seed", str(1_000_003 + a.seed)])):
        out = tmp / f"{name}.pt"
        print(f"--- {name} run: {' '.join(cmd + extra + ['--dump', str(out)])}", flush=True)
        r = subprocess.run(cmd + extra + ["--dump", str(out)])
        if r.returncode != 0:
            raise SystemExit(f"{name} run failed (exit {r.returncode})")
        dumps[name] = torch.load(out, weights_only=False)
    c, s = dumps["clean"], dumps["scrambled"]
    checks = {"losses": c["losses"] == s["losses"], "weights_sha256": c["weights_sha256"] == s["weights_sha256"]}
    if "val_codes" in c:
        checks["val_codes"] = c["val_codes"] is not None and torch.equal(c["val_codes"], s["val_codes"])
    ok = all(checks.values())
    verdict = {"command": " ".join(cmd), "steps": len(c["losses"]), "checks": checks, "pass": ok,
               "weights_sha256": c["weights_sha256"], "first_loss": c["losses"][0], "last_loss": c["losses"][-1]}
    print(json.dumps(verdict, indent=2))
    if a.report:
        with open(a.report, "a") as f:
            f.write(json.dumps(verdict) + "\n")
    print("FIREWALL TEST " + ("PASSED: identical with scrambled labels" if ok else "FAILED: the run depends on the labels"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
