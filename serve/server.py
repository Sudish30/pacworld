"""Playable Model 1 demo: FastAPI + WebSocket server.

  GET  /          the page (serve/static/index.html)
  WS   /ws        per-connection game loop: the browser sends key state when it
                  changes; the server ticks at a fixed rate, maps keys to an ALE
                  action, samples the next frame with the EMA diffusion model,
                  and sends it back as a binary PNG. JSON text messages carry
                  info/stats. Client -> server: {"type":"keys","keys":{"up":..}}
                  or {"type":"reset","ctx_sigma":0.0}.
  POST /reload    re-read the EMA checkpoint (pick up newer weights during training)
  GET  /status    loaded checkpoint step, clients, rolling fps / latency

Run:  python serve/server.py --seed 0 [--config configs/serve.yaml] [--port 8000]
"""
import argparse
import asyncio
import io
import json
import random
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
from common import load_config  # noqa: E402
from dataset import History, context_offsets, load_cache, load_split, to_uint8  # noqa: E402
from model1 import build_model, euler_sample  # noqa: E402

ACTION_NAMES = ["NOOP", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT"]


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
        _, self.val_idx = load_split(self.model_cfg, self.cache["ep_seed"])
        self.latencies = deque(maxlen=300)
        self.frame_times = deque(maxlen=300)
        self.clients = 0
        print(f"world ready: device={self.device} bf16={self.use_bf16} val episodes={len(self.val_idx)} "
              f"checkpoint step={self.step}")

    def load(self):
        path = ROOT / self.cfg["checkpoint"]
        ck = torch.load(path, map_location=self.device)
        model = build_model(ck["cfg"])
        model.load_state_dict(ck["ema"])
        model.eval().to(self.device)
        with self.lock:
            self.model, self.model_cfg, self.step = model, ck["cfg"], ck["step"]
            self.offsets = context_offsets(ck["cfg"]["data"])   # a reloaded checkpoint may use another context layout
        print(f"loaded {path} (training step {self.step})")
        return self.step

    def new_session(self, ctx_sigma=None):
        e = self.rng.choice(self.val_idx)
        a, b = int(self.cache["ep_start"][e]), int(self.cache["ep_start"][e + 1])
        start = min(self.cfg["start_step"], b - a - 2)
        # Seed the history with the real frames before `start`, back to the episode's first frame if the
        # context reaches that far, so the first steps of a session are clamped exactly as in training.
        hist = History.from_episodes([(self.cache["frames"][a:b], self.cache["actions"][a:b])], start, self.offsets, self.device)
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
        return to_uint8(pred[0]).permute(1, 2, 0).cpu().numpy()

    def stats(self):
        lat = sorted(self.latencies)
        ft = list(self.frame_times)
        fps = (len(ft) - 1) / (ft[-1] - ft[0]) if len(ft) > 1 and ft[-1] > ft[0] else 0.0
        p = lambda q: (lat[min(len(lat) - 1, int(q * len(lat)))] * 1000) if lat else 0.0
        return {"fps": round(fps, 1), "latency_p50_ms": round(p(0.5), 1), "latency_p95_ms": round(p(0.95), 1),
                "step": self.step, "clients": self.clients}


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
                         "ctx_sigma": world.cfg["ctx_sigma"]})


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

    async def send_info():
        await websocket.send_text(json.dumps({"type": "info", "seed": session.seed, "episode": session.episode,
                                              "step": world.step, "ctx_sigma": session.ctx_sigma,
                                              "upscale": world.cfg["upscale"], "fps_target": world.cfg["fps"]}))

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
