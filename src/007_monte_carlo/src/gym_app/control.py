import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


# Greedy policy w.r.t state-action values
def action(env: gym.Env, state, Q: NDArray, epsilon: float):
    if env.np_random.random() <= epsilon:
        return env.action_space.sample()
    action_values = Q[tuple(state)]
    max_val = np.max(action_values)
    best_a = np.flatnonzero(action_values == max_val)
    return env.np_random.choice(best_a)
