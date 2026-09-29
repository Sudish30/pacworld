"""Label every transition of the cache with the frozen LAM's code. Frames only (the labels are deleted at load).

codes[j] = the LAM code of transition frames[j] -> frames[j+1] (encoder frames at lam.encoder_offsets around target
j+1, clamped at the episode start by the one context rule); -1 on each episode's last frame. Same shape and meaning
as the cache's actions array, so data.action_source: lam can drop it into every window and rollout.
Writes data.lam_codes (int8 .npy) and a .json beside it (LAM checkpoint sha256, code counts, perplexity).

  python tools/lam_codes.py --config configs/lam-A.yaml --seed 0
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import load_config  # noqa: E402
from dataset import apply_action_source, codec_of, episode_end_markers, gather_context, load_cache  # noqa: E402
from lam import LAM, perplexity  # noqa: E402
from train_model0 import pick_device  # noqa: E402


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


@torch.no_grad()
def infer_codes(model, cache, cfg, device, batch_size, log=True):
    """(N,) int64 codes for every transition of the cache, -1 on each episode's last frame."""
    codec = codec_of(cache).to(device)
    frames = torch.from_numpy(np.asarray(cache["frames"]))  # memory-mapped for .npy caches
    starts = torch.from_numpy(cache["ep_start"])
    N = int(starts[-1])
    end = episode_end_markers(cache["ep_start"])
    targets = torch.from_numpy(np.nonzero(~end)[0] + 1)     # target i = j + 1 for every transition j
    off = torch.as_tensor(cfg["lam"]["encoder_offsets"], dtype=torch.long)
    codes = np.full(N, -1, np.int64)
    t0 = time.time()
    for b in range(0, len(targets), batch_size):
        t = targets[b:b + batch_size]
        first = starts[torch.searchsorted(starts, t, right=True) - 1]
        f, _, _ = gather_context(frames, frames[:, :1, :1], t, first, off)   # frames only; the dummy stands in for actions
        codes[(t - 1).numpy()] = model.encode(codec.decode(f.to(device))).cpu().numpy()
        if log and (b // batch_size) % 200 == 0:
            print(f"  {b + len(t)}/{len(targets)} transitions ({time.time() - t0:.0f}s)")
    return codes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=1024)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    cfg = load_config(ROOT / a.config)
    d = cfg["data"]
    device = pick_device(cfg["train"]["device"])
    ckpt = ROOT / cfg["train"]["checkpoint_dir"] / "lam.pt"
    ck = torch.load(ckpt, map_location=device)
    model = LAM(ck["cfg"]).to(device).eval()
    model.load_state_dict(ck["model"])
    cache = apply_action_source(load_cache(cfg, mmap=True), {**d, "action_source": "none"})   # labels deleted
    codes = infer_codes(model, cache, cfg, device, a.batch_size)
    out = ROOT / d["lam_codes"]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, codes.astype(np.int8))
    n = cfg["lam"]["n_codes"]
    counts = np.bincount(codes[codes >= 0], minlength=n)
    meta = {"lam_checkpoint": str(ckpt.relative_to(ROOT)), "lam_checkpoint_sha256": file_sha256(ckpt), "lam_step": ck["step"],
            "transitions": int((codes >= 0).sum()), "counts": counts.tolist(),
            "perplexity": perplexity(torch.from_numpy(counts)), "codes_sha256": hashlib.sha256(codes.astype(np.int8).tobytes()).hexdigest()}
    json.dump(meta, open(out.with_suffix(".json"), "w"), indent=2)
    print(f"wrote {out}: {meta['transitions']} transitions, counts {meta['counts']}, perplexity {meta['perplexity']:.2f}")


if __name__ == "__main__":
    main()
