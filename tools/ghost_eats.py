"""How many ghosts does the agent eat within one frightened phase? From recorded RAM (116) and rewards."""
import sys, glob, numpy as np
from multiprocessing import Pool
GHOST = (200, 400, 800, 1600); EXTRA = (0, 10, 50, 60, 100, 110)     # a ghost value plus a pellet / power pellet / cherry on the same step
def eats(r):
    r = int(r)
    return next((g for g in GHOST if r - g in EXTRA), 0)
def one(path):
    z = np.load(path); ram, rew = z["ram"], z["rewards"]; fr = ram[:, 116] > 0
    out, t, T = [], 1, len(fr)
    while t < T:
        if fr[t] and not fr[t - 1]:
            e = t
            while e < T and fr[e]: e += 1
            vals = [eats(x) for x in rew[t - 1:e]]; vals = [v for v in vals if v]
            out.append((len(vals), tuple(vals), e < T)); t = e
        else: t += 1
    return out
if __name__ == "__main__":
    total = {}
    for folder in sys.argv[1:]:
        with Pool(24) as pool: res = [p for ep in pool.map(one, sorted(glob.glob(folder + "/ep_*.npz")), chunksize=16) for p in ep]
        n = len(res); c = np.bincount([min(r[0], 5) for r in res], minlength=6)
        seq_ok = sum(1 for r in res if r[1] == GHOST[:len(r[1])])
        print(f"{folder}: {n} frightened phases; ghosts eaten per phase 0/1/2/3/4/5+: {c.tolist()} = {[f'{100 * x / n:.1f}%' for x in c]}; "
              f"mean {np.mean([r[0] for r in res]):.2f}; phases whose eat values follow 200,400,800,1600 in order: {100 * seq_ok / n:.1f}%")
        for k in range(6): total[k] = total.get(k, 0) + int(c[k])
    N = sum(total.values())
    print(f"ALL: {N} phases; 0/1/2/3/4/5+ ghosts: {[total[k] for k in range(6)]} = {[f'{100 * total[k] / N:.1f}%' for k in range(6)]}")
    print(f"training examples of the n-th ghost-eat in a phase: 1st {sum(total[k] for k in range(1, 6))}, 2nd {sum(total[k] for k in range(2, 6))}, 3rd {sum(total[k] for k in range(3, 6))}, 4th {sum(total[k] for k in range(4, 6))}")
