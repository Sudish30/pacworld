"""The 60-second explainer video for the README / LinkedIn (docs/pacworld_60s.mp4). No model is run here.

  1. title card
  2. the saved 450-step evaluation rollout used for docs/demo.gif: real game | 4-frame model | strided-context model,
     same start and recorded actions, at the game's speed, with the ghost pen outlined and a live count of how long
     the pen has been occupied (the same occupancy test as eval/pen_bootstrap.py)
  3. the one-frame experiment as a chart: parked starts out of 60 for the frame at -81 and at -80, original run and
     the four seeds of the registered replication (eval/results/timer/seeds_verdict.json)
  4. end card with the links

  python tools/make_video.py --config configs/video.yaml --seed 0
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from common import load_config  # noqa: E402
from dataset import episode_path  # noqa: E402
import detectors as D  # noqa: E402
from pen_bootstrap import run_longest  # noqa: E402
from pen_timer_analysis import geom  # noqa: E402


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size=size)


def centered(draw, y, text, f, fill, W):
    if draw.textlength(text, font=f) > W - 64:
        raise SystemExit(f"text too wide for the frame: {text!r}")
    draw.text(((W - draw.textlength(text, font=f)) / 2, y), text, font=f, fill=fill)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/video.yaml")
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    v = yaml.safe_load(open(ROOT / a.config))
    W, H, FPS, C, sec = v["width"], v["height"], v["fps"], {k: tuple(c) for k, c in v["colors"].items()}, v["seconds"]
    big, mid, small, bold = font(v["font_bold"], 54), font(v["font"], 30), font(v["font"], 22), font(v["font_bold"], 30)
    import imageio_ffmpeg
    writer = imageio_ffmpeg.write_frames(str(ROOT / v["out"]), (W, H), fps=FPS, codec="libx264", quality=8,
                                         macro_block_size=1, output_params=["-pix_fmt", "yuv420p"])
    writer.send(None)
    n_frames = 0

    def emit(img, seconds=None, frames=None):
        nonlocal n_frames
        data = np.asarray(img.convert("RGB")).tobytes()
        for _ in range(frames if frames is not None else round(seconds * FPS)):
            writer.send(data)
            n_frames += 1

    blank = lambda: Image.new("RGB", (W, H), C["bg"])

    # ---- 1. title
    img = blank()
    d = ImageDraw.Draw(img)
    centered(d, 230, "A neural network that draws Ms. Pac-Man", big, C["text"], W)
    centered(d, 310, "and loses track of time", big, C["accent"], W)
    centered(d, 420, "In the two right panels, every frame is drawn by the model, starting from real frames and replaying real key presses.", small, C["dim"], W)
    emit(img, sec["title"])

    # ---- 2. the rollout: the same rule and files as tools/make_demo_gif.py
    c = yaml.safe_load(open(ROOT / v["pen_stats"]))
    g = c["demo_gif"]
    pick = c["runs"][g["pick"]["run"]]
    ecfg = load_config(ROOT / pick["config"])
    _, ml, _ = run_longest(ecfg, ROOT / pick["dir"], g["steps"])
    row = int(np.nonzero(ml >= g["pick"]["min_longest"])[0][0])
    job = list(csv.DictReader(open(ROOT / pick["dir"] / "jobs.csv")))[row]
    start = ecfg["rollout"]["start_step"]
    native = np.load(episode_path(ecfg["data"], int(job["episode"]), ROOT))["frames"][start:start + g["steps"]]
    panels = [("the real game", D.downsample_frames(native, *D.frame_geometry(ecfg)))]
    names = {"ctx4": "model that sees its last 4 frames", "ft-uniform": "model that also sees 6 older frames"}
    for name in g["panels"]:
        j2 = list(csv.DictReader(open(ROOT / c["runs"][name]["dir"] / "jobs.csv")))[row]
        if (j2["episode"], j2["seed"]) != (job["episode"], job["seed"]):
            raise SystemExit(f"{name}: jobs.csv row {row} is a different rollout")
        panels.append((names.get(name, name), np.load(ROOT / c["runs"][name]["dir"] / "preds_all.npy", mmap_mode="r")[row, :g["steps"]]))
    ref = D.load_reference(ecfg)
    G = geom(ref.size)
    box, bg = G["pen_box"], ref.bg64[G["pen_box"]]
    occ = lambda fr: (np.linalg.norm(np.asarray(fr)[:, box[0], box[1]].astype(float) - bg[None], axis=-1) > G["occ_resid"]).sum((1, 2)) >= G["occ_pixels"]
    runs = []
    for _, fr in panels:                              # consecutive occupied steps up to each frame
        o, cur, out = occ(fr), 0, []
        for x in o:
            cur = cur + 1 if x else 0
            out.append(cur)
        runs.append(out)
    S, pad = 384, 32
    x0 = (W - (3 * S + 2 * pad)) // 2
    px0, py0, px1, py1 = [t * S // 64 for t in v["pen_box_64"]]
    reps = FPS // v["game_fps"]
    T = min(g["steps"], sec["rollout"] * v["game_fps"])
    for t in range(T):
        img = blank()
        d = ImageDraw.Draw(img)
        centered(d, 36, "Same start, same key presses, 30 seconds", mid, C["text"], W)
        for k, (label, fr) in enumerate(panels):
            x = x0 + k * (S + pad)
            img.paste(Image.fromarray(np.asarray(fr[t]).astype(np.uint8)).resize((S, S), Image.NEAREST), (x, 130))
            d.rectangle([x + px0 - 3, 130 + py0 - 3, x + px1 + 3, 130 + py1 + 3], outline=C["accent"], width=3)
            d.text((x + (S - d.textlength(label, font=small)) / 2, 96), label, font=small, fill=C["text"])
            n = runs[k][t]
            txt = f"pen occupied for {n} steps" if n else "pen empty"
            col = C["bad"] if n > v["real_max_stay"] else C["dim"]
            d.text((x + (S - d.textlength(txt, font=small)) / 2, 130 + S + 14), txt, font=small, fill=col)
        centered(d, 596, "The yellow box is the ghost pen. In the real game a ghost waits there at most 91 steps, with no countdown on screen.", small, C["dim"], W)
        centered(d, 626, "The 4-frame model cannot tell how long a ghost has waited, so it leaves one parked. Both models drift from the real game.", small, C["dim"], W)
        centered(d, 656, "A picked example of the failure. Over 30 test rollouts, the 4-frame model parks a ghost 200+ steps in 17; the other in 0.", small, C["accent"], W)
        d.text((x0, 690), f"step {t + 1} of {T}", font=small, fill=C["dim"])
        emit(img, frames=reps)

    # ---- 3. the one-frame experiment
    sv = json.load(open(ROOT / v["seeds_verdict"]))["cells"]
    parked = lambda cell: [round(60 * (1 - sv[cell][s]["release"])) for s in sorted(sv[cell], key=int)]
    a81, a80 = parked("S10:80"), parked("S10b:80")
    total = round(sec["experiment"] * FPS)
    for i in range(total):
        img = blank()
        d = ImageDraw.Draw(img)
        centered(d, 36, "Why? A small test game with a hidden 80-step wait", mid, C["text"], W)
        centered(d, 84, "Each model sees 10 past frames. The two rows differ only in ONE of them, moved by a single step.", small, C["dim"], W)
        grow = min(1.0, i / (0.35 * total))
        for k, (title, vals, col, y) in enumerate((("one past frame sits 81 steps back", a81, C["bad"], 170),
                                                   ("one past frame sits exactly 80 steps back", a80, C["good"], 400))):
            d.text((120, y), title, font=bold, fill=C["text"])
            for j, val in enumerate(vals):
                bx = 120 + j * 210
                h = int(130 * val / 60 * grow)
                d.rectangle([bx, y + 60 + 130 - h, bx + 150, y + 60 + 130], fill=col)
                d.rectangle([bx, y + 60, bx + 150, y + 60 + 130], outline=C["dim"], width=1)
                lab = f"{round(val * grow)} of 60 parked"
                d.text((bx + (150 - d.textlength(lab, font=small)) / 2, y + 60 + 136), lab, font=small, fill=C["text"])
                run = "original" if j == 0 else f"retrain {j}"
                d.text((bx + (150 - d.textlength(run, font=small)) / 2, y + 34), run, font=small, fill=C["dim"])
        if i > 0.45 * total:
            centered(d, 636, "In this test game, a model timed the hidden wait when one of its past frames sat exactly that far back.", small, C["accent"], W)
        if i > 0.6 * total:
            centered(d, 666, "The 80-step fix was proposed after the first run parked, then registered before it ran; so were the 4 retrains.", small, C["dim"], W)
            centered(d, 694, "A related registered check (short gaps are harmless) failed, and is reported.", small, C["dim"], W)
        emit(img, frames=1)

    # ---- 4. end card
    img = blank()
    d = ImageDraw.Draw(img)
    centered(d, 200, "Play it in your browser", big, C["text"], W)
    centered(d, 290, v["demo_url"], big, C["accent"], W)
    centered(d, 362, "Desktop Chrome or Edge with a keyboard (WebGPU).", small, C["dim"], W)
    centered(d, 400, f"code, results and paper draft: {v['repo_url']}", mid, C["text"], W)
    centered(d, 480, "A research demo. The model drifts, and several things we tried failed; the write-up says which.", small, C["dim"], W)
    centered(d, 512, "Not affiliated with Bandai Namco or Atari.", small, C["dim"], W)
    emit(img, sec["end"])
    writer.close()
    out = ROOT / v["out"]
    print(f"rollout row {row} (episode {job['episode']}, sampler seed {job['seed']}); parked of 60 at -81: {a81}, at -80: {a80}; "
          f"{n_frames} frames = {n_frames / FPS:.1f} s -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
