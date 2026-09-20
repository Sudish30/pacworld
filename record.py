"""Record Ms. Pac-Man episodes for world-model training.

Each finished episode is written to data/{mode}/ep_{seed}.npz with:
  frames      uint8 (T+1, H, W, 3)  cropped RGB observations obs_0 .. obs_T
  actions     int64 (T,)            actions[i] was taken after observing frames[i]
  rewards     float32 (T,)
  terminated  bool (T,)             terminated[i] is the flag returned by step i
  ram         uint8 (T+1, 128)      emulator RAM alongside each frame
  seed        int                   env.reset(seed=...) that reproduces the episode
"""
import argparse
from collections import deque
from pathlib import Path
import time

import numpy as np
from PIL import Image

from common import crop, get_ram, load_config, make_env, skip_step


class PPOAgent:
    """Wraps the sb3 zoo PPO checkpoint with the 84x84 grayscale 4-frame view it expects."""

    def __init__(self, cfg, n_envs):
        from huggingface_hub import hf_hub_download
        import gymnasium as gym
        from stable_baselines3 import PPO

        a = cfg["agent"]
        self.size = a["obs_size"]
        self.k = a["frame_stack"]
        path = hf_hub_download(a["hf_repo"], a["hf_file"])
        # The zip was pickled against legacy gym; override the spaces and schedules.
        custom = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
            "observation_space": gym.spaces.Box(0, 255, (self.k, self.size, self.size), np.uint8),
            "action_space": gym.spaces.Discrete(9),
        }
        self.model = PPO.load(path, device=a["device"], custom_objects=custom)
        self.stacks = [deque(maxlen=self.k) for _ in range(n_envs)]

    def _warp(self, frame):
        img = Image.fromarray(frame).convert("L").resize((self.size, self.size), Image.BOX)
        return np.asarray(img, dtype=np.uint8)

    def reset(self, i, frame):
        self.stacks[i].clear()
        for _ in range(self.k - 1):
            self.stacks[i].append(np.zeros((self.size, self.size), np.uint8))
        self.stacks[i].append(self._warp(frame))

    def observe(self, i, frame):
        self.stacks[i].append(self._warp(frame))

    def act(self, indices):
        obs = np.stack([np.stack(self.stacks[i]) for i in indices])
        actions, _ = self.model.predict(obs, deterministic=False)
        return actions


class PelletSeeker:
    """Shortest-path walker to the nearest remaining power pellet, on a maze graph mined from recorded RAM."""

    def __init__(self, scfg, n_envs, rng):
        self.c, self.rng = scfg, rng
        g = np.load(scfg["graph"])
        self.out = {}
        for x0, y0, x1, y1 in g["edges"].astype(int):
            self.out.setdefault((x0, y0), []).append((x1, y1))
        self.nodes = np.array(sorted(set(self.out) | {n for v in self.out.values() for n in v}))
        rev = {}
        for a, nbrs in self.out.items():
            for b in nbrs:
                rev.setdefault(b, []).append(a)
        self.dist = {}                                    # pellet name -> {position: steps to the pellet}
        for name, (px, py) in scfg["power_pellets"].items():
            goal = [tuple(n) for n in self.nodes if abs(n[0] - px) + abs(n[1] - py) <= scfg["pellet_radius"]]
            d, frontier = {n: 0 for n in goal}, list(goal)
            while frontier:
                nxt = []
                for b in frontier:
                    for a in rev.get(b, []):
                        if a not in d:
                            d[a] = d[b] + 1
                            nxt.append(a)
                frontier = nxt
            self.dist[name] = d
        self.remaining = [set(scfg["power_pellets"]) for _ in range(n_envs)]
        self.active = [True] * n_envs
        self.stats = {"seek_steps": 0, "ppo_steps": 0}

    def reset(self, i):
        self.remaining[i] = set(self.c["power_pellets"])
        self.active[i] = self.rng.random() < self.c["episode_prob"]

    def observe(self, i, ram, reward):
        if reward == self.c["power_pellet_reward"]:
            x, y = int(ram[self.c["pac_x_ram"]]), int(ram[self.c["pac_y_ram"]])
            if not self.remaining[i]:                      # a new level refills the maze
                self.remaining[i] = set(self.c["power_pellets"])
            near = min(self.remaining[i], key=lambda n: abs(self.c["power_pellets"][n][0] - x) + abs(self.c["power_pellets"][n][1] - y))
            self.remaining[i].discard(near)

    def action(self, i, ram):
        """Cardinal action toward the nearest remaining power pellet, or None to let the PPO agent act."""
        c = self.c
        if not self.active[i] or not self.remaining[i] or ram[c["frightened_ram"]] > 0:
            return None
        x, y = int(ram[c["pac_x_ram"]]), int(ram[c["pac_y_ram"]])
        gx, gy = ram[c["ghost_x_ram"]].astype(int), ram[c["ghost_y_ram"]].astype(int)
        if (np.abs(gx - x) + np.abs(gy - y)).min() < c["ghost_avoid_dist"]:
            return None
        pos = (x, y)
        if pos not in self.out:
            d = np.abs(self.nodes[:, 0] - x) + np.abs(self.nodes[:, 1] - y)
            j = int(d.argmin())
            if d[j] > c["snap_radius"]:
                return None
            pos = tuple(self.nodes[j])
        best = None
        for name in self.remaining[i]:
            for b in self.out.get(pos, []):
                db = self.dist[name].get(b)
                if db is not None and (best is None or db < best[0]):
                    best = (db, b)
        if best is None:
            return None
        dx, dy = best[1][0] - x, best[1][1] - y
        if abs(dx) >= abs(dy):
            return 2 if dx > 0 else 3                      # RIGHT / LEFT
        return 4 if dy > 0 else 1                          # DOWN / UP (RAM y grows downwards)


class EpisodeBuffer:
    def __init__(self, seed, first_frame, first_ram):
        self.seed = seed
        self.frames = [first_frame]
        self.ram = [first_ram]
        self.actions, self.rewards, self.terminated = [], [], []

    def step(self, action, reward, terminated, frame, ram):
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminated.append(terminated)
        self.frames.append(frame)
        self.ram.append(ram)

    def save(self, out_dir, action_meanings):
        path = out_dir / f"ep_{self.seed}.npz"
        np.savez_compressed(
            path,
            frames=np.stack(self.frames).astype(np.uint8),
            actions=np.asarray(self.actions, dtype=np.int64),
            rewards=np.asarray(self.rewards, dtype=np.float32),
            terminated=np.asarray(self.terminated, dtype=bool),
            ram=np.stack(self.ram).astype(np.uint8),
            seed=np.int64(self.seed),
            action_meanings=np.asarray(action_meanings),
        )
        return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/record.yaml")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--mode", choices=["agent", "random"])
    p.add_argument("--steps", type=int, help="total env steps across all envs")
    p.add_argument("--n-envs", type=int)
    p.add_argument("--eps", type=float)
    p.add_argument("--sticky-prob", type=float)
    p.add_argument("--out-dir")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    for key in ["mode", "steps", "n_envs", "eps", "sticky_prob", "out_dir"]:
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)

    mode, n_envs, budget = cfg["mode"], cfg["n_envs"], cfg["steps"]
    out_dir = Path(cfg["out_dir"]) / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    envs = [make_env(cfg) for _ in range(n_envs)]
    n_actions = envs[0].action_space.n
    action_meanings = envs[0].unwrapped.get_action_meanings()
    agent = PPOAgent(cfg, n_envs) if mode == "agent" else None
    seeker = PelletSeeker(cfg["seek"], n_envs, rng) if mode == "agent" and (cfg.get("seek") or {}).get("enabled") else None
    lo, hi = cfg["sticky_len"]

    def new_seed():
        while True:
            s = int(rng.integers(0, 2**31 - 1))
            if not (out_dir / f"ep_{s}.npz").exists():
                return s

    def start_episode(i):
        s = new_seed()
        obs, _ = envs[i].reset(seed=s)
        frame = crop(obs, cfg)
        if agent:
            agent.reset(i, frame)
        buffers[i] = EpisodeBuffer(s, frame, get_ram(envs[i]))
        sticky[i] = (0, 0)
        if seeker:
            seeker.reset(i)

    buffers = [None] * n_envs
    sticky = [(0, 0)] * n_envs  # (remaining steps, action)
    for i in range(n_envs):
        start_episode(i)

    steps_done, episodes, returns = 0, 0, []
    t0 = time.time()
    print(f"mode={mode} n_envs={n_envs} budget={budget} seed={args.seed} -> {out_dir}")

    while any(b is not None for b in buffers):
        active = [i for i in range(n_envs) if buffers[i] is not None]
        actions = {}
        if agent:
            policy_actions = dict(zip(active, agent.act(active)))
        for i in active:
            if mode == "random":
                actions[i] = int(rng.integers(n_actions))
                continue
            remaining, a = sticky[i]
            if remaining > 0:
                sticky[i] = (remaining - 1, a)
                actions[i] = a
            elif rng.random() < cfg["sticky_prob"]:
                a = int(rng.integers(n_actions))
                sticky[i] = (int(rng.integers(lo, hi + 1)) - 1, a)
                actions[i] = a
            elif rng.random() < cfg["eps"]:
                actions[i] = int(rng.integers(n_actions))
            else:
                a_seek = seeker.action(i, buffers[i].ram[-1]) if seeker else None
                actions[i] = int(policy_actions[i]) if a_seek is None else a_seek
                if seeker:
                    seeker.stats["ppo_steps" if a_seek is None else "seek_steps"] += 1

        for i in active:
            obs, reward, terminated, truncated = skip_step(envs[i], actions[i], cfg)
            frame = crop(obs, cfg)
            buffers[i].step(actions[i], reward, terminated, frame, get_ram(envs[i]))
            if seeker:
                seeker.observe(i, buffers[i].ram[-1], reward)
            if agent:
                agent.observe(i, frame)
            steps_done += 1
            if terminated or truncated:
                path = buffers[i].save(out_dir, action_meanings)
                ep_len, ep_ret = len(buffers[i].actions), float(np.sum(buffers[i].rewards))
                episodes += 1
                returns.append(ep_ret)
                print(f"[{steps_done:>7}/{budget}] ep {episodes:>4} len={ep_len:>5} return={ep_ret:>7.0f} -> {path.name}")
                buffers[i] = None
                if steps_done < budget:
                    start_episode(i)

    dt = time.time() - t0
    print(f"done: {episodes} episodes, {steps_done} steps in {dt/60:.1f} min ({steps_done/dt:.0f} steps/s), "
          f"mean return {np.mean(returns):.0f}")
    if seeker:
        print(f"seeking: {seeker.stats['seek_steps']} policy steps walked toward a power pellet, {seeker.stats['ppo_steps']} left to the PPO agent")


if __name__ == "__main__":
    main()
