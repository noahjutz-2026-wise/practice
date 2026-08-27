import numpy as np
from numpy.typing import NDArray

"""
First-Visit MC prediction
Given one episode
"""


def monte_carlo(
    rewards: NDArray,
    states: NDArray,
    actions: NDArray,
    gamma: float,
    M: NDArray,
    Q: NDArray,
) -> tuple[NDArray, NDArray]:
    T = len(rewards)
    # Build set of first-visit timesteps: for each (state, action) pair,
    # only the earliest timestep counts.
    first_visit: dict[tuple, int] = {}
    for t in range(T):
        key = (*states[t], actions[t])
        if key not in first_visit:
            first_visit[key] = t

    # Backward pass to compute returns
    G = 0.0
    for t in range(T - 1, -1, -1):
        G = gamma * G + rewards[t]
        key = (*states[t], actions[t])
        if first_visit[key] == t:
            idx = tuple(key)
            M[idx] += 1
            Q[idx] += (G - Q[idx]) / M[idx]
    return M, Q
