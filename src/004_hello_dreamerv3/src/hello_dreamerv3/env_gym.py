from typing import SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType
from typing_extensions import override


class MyGymEnv(gym.Env[np.ndarray, int]):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=0, high=200, shape=(1,))
        self.action_space = gym.spaces.Discrete(11)

        self.render_mode = "rgb_array"

        self._state: int = 0

    @override
    def step(
        self, action: int
    ) -> tuple[float, SupportsFloat, bool, bool, dict[str, str]]:

        reward: int = action - 1
        self._state = max(0, min(200, self._state + reward))
        terminated: bool = self._state >= 200
        truncated = False

        observation = self._get_obs()

        return observation, reward, terminated, truncated, {}

    def _get_obs(self) -> tuple[float]:
        return (self._state,)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[ObsType, dict]:
        self._state = 0
        return self.observation_space.sample(), {}
