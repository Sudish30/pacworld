"""Verify that ALE/MsPacman-v5 loads and runs."""
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5")
obs, info = env.reset(seed=0)
print("observation shape:", obs.shape, "dtype:", obs.dtype)
print("action space:", env.action_space)
print("action meanings:", env.unwrapped.get_action_meanings())
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print("one step ok -> reward:", reward, "terminated:", terminated, "truncated:", truncated)
env.close()
