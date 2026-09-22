"""Playable Model 1 demo: FastAPI + WebSocket server.

  GET  /          the page (serve/static/index.html)
  WS   /ws        per-connection game loop: the browser sends key state when it
                  changes; the server ticks at a fixed rate, maps keys to an ALE
                  action, samples the next frame with the EMA diffusion model,
                  and sends it back as a binary PNG. JSON text messages carry
                  info/stats. Client -> server: {"type":"keys","keys":{"up":..}}
                  or {"type":"reset","ctx_sigma":0.0}.
  POST /reload    re-read the EMA checkpoint (pick up newer weights during training)
  GET  /status    loaded checkpoint step and lineage, clients, rolling fps / latency, watchdog resets

Watchdog: the model can lose Pac-Man for good (an empty maze after a death, an endless frightened phase), which
leaves nothing to control. The real game never hides him for a single frame, so when the detector finds no
Pac-Man on `missing_checks` consecutive checks the server starts a fresh session and tells the page why.

Run:  python serve/server.py --seed 0 [--config configs/serve.yaml] [--port 8000]
"""
import argparse
import asyncio
import io
import json
import random
import re
import statistics
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from common import load_config  # noqa: E402
from dataset import FrameCodec, History, context_offsets, load_cache, load_split, to_uint8  # noqa: E402
from model1 import build_model, euler_sample  # noqa: E402
import detectors as D  # noqa: E402

ACTION_NAMES = ["NOOP", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT"]


def _short(run_dir, parent_dir=None):
    """m1-2M-ctx6s16-ft-uniform -> ft-uniform (relative to its parent m1-2M-ctx6s16) or ctx6s16 (no parent)."""
    name = Path(run_dir).name
    if parent_dir and name.startswith(Path(parent_dir).name + "-"):
        return name[len(Path(parent_dir).name) + 1:]
    return re.sub(r"^m1-\d+M-", "", name) or name


def _k(steps):
    return f"{steps // 1000}k" if steps % 1000 == 0 else str(steps)


def lineage(path, ck=None, depth=0):
    """'ft-uniform, 15k steps from ctx6s16@100k' - follows finetune.init_from through the parent checkpoints."""
    path = Path(path)
    ck = ck or torch.load(path, map_location="cpu", mmap=True)
    init = (ck["cfg"].get("finetune") or {}).get("init_from")
    if not init or depth > 4 or not (ROOT / init).exists():
        return f"{_short(path.parent if path.parent.name != 'checkpoints' else path.stem)}, {_k(ck['step'])} steps from scratch"
    parent = torch.load(ROOT / init, map_location="cpu", mmap=True)
    here = f"{_short(path.parent, Path(init).parent)}, {_k(ck['step'])} steps from {_short(Path(init).parent)}@{_k(parent['step'])}"
    grand = (parent["cfg"].get("finetune") or {}).get("init_from")
    return here + (f"; {lineage(ROOT / init, parent, depth + 1)}" if grand else "")


def keys_to_action(keys):
    up = keys.get("up") and not keys.get("down")
    down = keys.get("down") and not keys.get("up")
    left = keys.get("left") and not keys.get("right")
    right = keys.get("right") and not keys.get("left")
    if up and right:
        return 5
    if up and left:
        return 6
    if down and right:
        return 7
    if down and left:
        return 8
    if up:
        return 1
    if right:
        return 2
    if left:
        return 3
    if down:
        return 4
    return 0


class Session:
    def __init__(self, hist, seed, episode, ctx_sigma):
        self.hist = hist            # dataset.History: past frames and actions; builds each context with the training rule
        self.seed = seed
        self.episode = episode
        self.ctx_sigma = ctx_sigma
        self.frames = 0
        self.missing = 0            # consecutive watchdog checks without Pac-Man


class World:
    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bf16 = cfg["bf16"] and self.device.type == "cuda"
        self.rng = random.Random(seed)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.lock = threading.Lock()
        self.model, self.model_cfg, self.step = None, None, None
        self.load()
        self.dcfg = self.model_cfg["diffusion"]
        self.cache = load_cache(self.model_cfg, mmap=True)   # only a few val frames are read per session
        self.codec = FrameCodec(self.cache.get("palette"))   # a palette cache stores indices; History works in RGB
        _, self.val_idx = load_split(self.model_cfg, self.cache["ep_seed"])
        self.wd = cfg.get("watchdog") or {}
        self.ref = self._reference() if self.wd.get("enabled") else None
        self.auto_resets = 0
        self.latencies = deque(maxlen=300)
        self.frame_times = deque(maxlen=300)
        self.clients = 0
        print(f"world ready: device={self.device} bf16={self.use_bf16} val episodes={len(self.val_idx)} "
              f"checkpoint step={self.step}")

    def _reference(self):
        """Watchdog detector built for the served model's own frame size and resample filter."""
        dcfg = load_config(ROOT / self.wd["detector_config"])
        dcfg["data"]["size"] = self.model_cfg["data"]["size"]
        dcfg["data"]["resample"] = self.model_cfg["data"].get("resample", "box")
        dcfg["detector"]["min_blob_weight"] *= (dcfg["data"]["size"] / 64) ** 2
        return D.load_reference(dcfg)

    def load(self):
        path = ROOT / self.cfg["checkpoint"]
        ck = torch.load(path, map_location=self.device)
        model = build_model(ck["cfg"])
        model.load_state_dict(ck["ema"])
        model.eval().to(self.device)
        with self.lock:
            self.model, self.model_cfg, self.step = model, ck["cfg"], ck["step"]
            self.offsets = context_offsets(ck["cfg"]["data"])   # a reloaded checkpoint may use another context layout
            self.lineage = lineage(path, ck)
        print(f"loaded {path} (training step {self.step}; {self.lineage})")
        return self.step

    def new_session(self, ctx_sigma=None):
        e = self.rng.choice(self.val_idx)
        a, b = int(self.cache["ep_start"][e]), int(self.cache["ep_start"][e + 1])
        start = min(self.cfg["start_step"], b - a - 2)
        # Seed the history with the real frames before `start`, back to the episode's first frame if the
        # context reaches that far, so the first steps of a session are clamped exactly as in training.
        hist = History.from_episodes([(self.codec.decode_np(self.cache["frames"][a:b]), self.cache["actions"][a:b])],
                                     start, self.offsets, self.device)
        sigma = self.cfg["ctx_sigma"] if ctx_sigma is None else float(ctx_sigma)
        return Session(hist, int(self.cache["ep_seed"][e]), e, sigma)

    @torch.inference_mode()
    def predict(self, session, action):
        """Sample the next frame for `action`, advance the session, return HxWx3 uint8."""
        ctx, acts = session.hist.context(torch.tensor([action], device=self.device))
        sigma = torch.full((1,), session.ctx_sigma, device=self.device)
        if session.ctx_sigma > 0:
            ctx = ctx + session.ctx_sigma * torch.randn(ctx.shape, device=self.device, generator=self.gen)
        with self.lock:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                pred = euler_sample(self.model, ctx, acts, sigma, self.cfg["sampler_steps"], self.dcfg, self.gen)
        pred = pred.float()
        session.hist.push(pred)
        session.frames += 1
        frame = to_uint8(pred[0]).permute(1, 2, 0).cpu().numpy()
        if self.ref is not None and session.frames % self.wd["check_every"] == 0:
            session.missing = 0 if self.ref.sprites(frame)["pac"] is not None else session.missing + 1
        return frame

    def lost(self, session):
        """True when Pac-Man has been missing for watchdog.missing_checks consecutive checks."""
        return self.ref is not None and session.missing >= self.wd["missing_checks"]

    def stats(self):
        lat = sorted(self.latencies)
        ft = list(self.frame_times)
        fps = (len(ft) - 1) / (ft[-1] - ft[0]) if len(ft) > 1 and ft[-1] > ft[0] else 0.0
        p = lambda q: (lat[min(len(lat) - 1, int(q * len(lat)))] * 1000) if lat else 0.0
        return {"fps": round(fps, 1), "latency_p50_ms": round(p(0.5), 1), "latency_p95_ms": round(p(0.95), 1),
                "step": self.step, "clients": self.clients, "auto_resets": self.auto_resets}


def encode_png(frame):
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


app = FastAPI()
world: World = None  # set in main()
STATIC = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/status")
async def status():
    return JSONResponse({**world.stats(), "checkpoint": world.cfg["checkpoint"], "fps_target": world.cfg["fps"],
                         "sampler_steps": world.cfg["sampler_steps"], "context_offsets": world.offsets,
                         "ctx_sigma": world.cfg["ctx_sigma"], "lineage": world.lineage,
                         "watchdog": world.wd if world.wd.get("enabled") else None})


@app.post("/reload")
async def reload():
    step = await asyncio.to_thread(world.load)
    return JSONResponse({"ok": True, "step": step})


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    world.clients += 1
    session = world.new_session()
    keys = {}
    period = 1.0 / world.cfg["fps"]

    async def send_info(reason=None):
        await websocket.send_text(json.dumps({"type": "info", "seed": session.seed, "episode": session.episode,
                                              "step": world.step, "ctx_sigma": session.ctx_sigma, "lineage": world.lineage,
                                              "upscale": world.cfg["upscale"], "fps_target": world.cfg["fps"], "reason": reason}))

    async def receiver():
        nonlocal session
        while True:
            data = json.loads(await websocket.receive_text())
            if data.get("type") == "keys":
                keys.clear()
                keys.update(data.get("keys", {}))
            elif data.get("type") == "reset":
                session = world.new_session(data.get("ctx_sigma"))
                await send_info()

    recv_task = asyncio.create_task(receiver())
    await send_info()
    next_tick = time.perf_counter()
    last_stats = last_log = time.perf_counter()
    try:
        while True:
            t0 = time.perf_counter()
            action = keys_to_action(keys)
            frame = await asyncio.to_thread(world.predict, session, action)
            await websocket.send_bytes(encode_png(frame))
            if world.lost(session):
                steps = world.wd["missing_checks"] * world.wd["check_every"]
                world.auto_resets += 1
                print(f"[watchdog] Pac-Man missing for {steps} steps at session frame {session.frames}: new session")
                session = world.new_session(session.ctx_sigma)
                await send_info(f"Pac-Man was lost for {steps / world.cfg['fps']:.0f} s - new board")
            now = time.perf_counter()
            world.latencies.append(now - t0)
            world.frame_times.append(now)
            if now - last_stats >= 1.0:
                last_stats = now
                await websocket.send_text(json.dumps({"type": "stats", **world.stats(), "action": ACTION_NAMES[action],
                                                      "frames": session.frames}))
            if now - last_log >= world.cfg["stats_log_every_s"]:
                last_log = now
                s = world.stats()
                print(f"[stats] fps {s['fps']:5.1f}  latency p50 {s['latency_p50_ms']:5.1f} ms  "
                      f"p95 {s['latency_p95_ms']:5.1f} ms  clients {s['clients']}  action {ACTION_NAMES[action]}")
            next_tick += period
            delay = next_tick - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                next_tick = time.perf_counter()  # running late: don't try to catch up
            if recv_task.done():
                recv_task.result()  # raises WebSocketDisconnect if the client went away
    except WebSocketDisconnect:
        pass
    finally:
        recv_task.cancel()
        world.clients -= 1


def main():
    global world
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/serve.yaml")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--checkpoint")
    args = p.parse_args()
    cfg = load_config(args.config)
    for k in ("host", "port", "checkpoint"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    world = World(cfg, args.seed)
    uvicorn.run(app, host=cfg["host"], port=cfg["port"], log_level="warning")


if __name__ == "__main__":
    main()
