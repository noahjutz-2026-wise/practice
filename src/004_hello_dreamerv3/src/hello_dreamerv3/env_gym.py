from typing import SupportsFloat

import gymnasium as gym
import numpy as np
from typing_extensions import override


class MyGymEnv(gym.Env[np.ndarray, int]):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=0, high=200)
        self.action_space = gym.spaces.Discrete(11)

        self.render_mode = "rgb_array"

        self._state: int = 0

    @override
    def step(
        self, action: int
    ) -> tuple[float, SupportsFloat, bool, bool, dict[str, str]]:

        reward: int = action - 1
        self._state += reward
        terminated: bool = self._state >= 200
        truncated = False

        observation = self._get_obs()

        return observation, reward, terminated, truncated, {}

    def _get_obs(self) -> float:
        return self._state

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._state = 0
