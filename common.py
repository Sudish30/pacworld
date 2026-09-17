"""Shared helpers: config loading, env construction, frame cropping."""
import os
from pathlib import Path

# Keep the Hugging Face cache inside the project and use plain HTTPS downloads.
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import gymnasium as gym
import ale_py
import numpy as np
import yaml

gym.register_envs(ale_py)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def make_env(cfg):
    e = cfg["env"]
    return gym.make(
        e["id"],
        frameskip=e["frameskip"],
        repeat_action_probability=e["repeat_action_probability"],
        full_action_space=e["full_action_space"],
    )


def crop(frame, cfg):
    c = cfg["crop"]
    return frame[c["top"]:c["bottom"]]


def get_ram(env):
    return np.asarray(env.unwrapped.ale.getRAM(), dtype=np.uint8)
