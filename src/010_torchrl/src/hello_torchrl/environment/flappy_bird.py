import gymnasium as gym
import numpy as np
from ray.rllib.env.wrappers.atari_wrappers import NormalizedImageEnv

from hello_torchrl.util.flappy_bird_patch import (
    flappy_bird_gymnasium,  # pyright: ignore[reportUnusedImport]  # noqa: F401
)

env: gym.Env[gym.spaces.Box, gym.spaces.Discrete[np.uint8]] = gym.make(
    "FlappyBird-v0", render_mode="rgb_array", audio_on=False
)
env = gym.wrappers.AddRenderObservation(env, render_only=True)
env = gym.wrappers.ResizeObservation(env, (64, 64))
env = NormalizedImageEnv(env)
