import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


# Greedy policy w.r.t state-action values
def action(env: gym.Env, state, Q: NDArray):
    action_values = Q[tuple(state)]
    a = np.argmax(action_values)
    return a
