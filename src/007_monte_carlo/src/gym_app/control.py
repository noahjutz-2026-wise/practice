import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


# Greedy policy w.r.t state-action values
def action(env: gym.Env, state, Q: NDArray):
    action_values = Q[tuple(state)]
    max_val = np.max(action_values)
    best_a = np.flatnonzero(action_values == max_val)
    return np.random.choice(best_a)
