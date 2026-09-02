from typing import cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


def main():
    env = cast(gym.Env[NDArray[np.float64], int], gym.make("LunarLander-v3"))  # pyright: ignore[reportUnknownMemberType]
    _observation, _info = env.reset()
    print("_____OBSERVATION SPACE_____ \n")
    print("Observation Space Shape", env.observation_space.shape)
    print(
        "Sample observation", env.observation_space.sample()
    )  # Get a random observation
    for _ in range(20):
        action = env.action_space.sample()
        _observation, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            _observation, _info = env.reset()

    env.close()
