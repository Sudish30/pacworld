"""Count frightened phases (RAM 116) in a folder of recorded episodes: starts, ends, per episode, per 1000 steps.

  python tools/count_phases.py data/pellet/agent
"""
import sys, glob, numpy as np
files = sorted(glob.glob(sys.argv[1] + "/ep_*.npz"))
ends = starts = steps = deaths = 0; per_ep = []; where = {}
for f in files:
    ram = np.load(f)["ram"]; fr = ram[:, 116] > 0
    e = int((fr[:-1] & ~fr[1:]).sum()); s = np.nonzero(~fr[:-1] & fr[1:])[0] + 1
    ends += e; starts += len(s); steps += len(ram) - 1; per_ep.append(e)
    deaths += int((np.diff(ram[:, 123].astype(int)) < 0).sum())
    for t in s:
        k = ("top" if ram[t, 16] < 80 else "bottom") + ("_left" if ram[t, 10] < 88 else "_right"); where[k] = where.get(k, 0) + 1
print(f"{sys.argv[1]}: {len(files)} episodes, {steps} steps, mean length {steps / max(len(files), 1):.0f}; phase starts {starts}, phase ENDS {ends} "
      f"= {ends / max(len(files), 1):.2f} per episode = {1000 * ends / max(steps, 1):.2f} per 1000 steps; episodes with 0/1/2/3/4+ ends: {[sum(1 for x in per_ep if x == k) for k in range(4)] + [sum(1 for x in per_ep if x >= 4)]}; which pellet: {where}")
