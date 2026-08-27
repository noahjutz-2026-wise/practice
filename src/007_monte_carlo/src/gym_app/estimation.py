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
    G = 0
    for t in range(len(rewards) - 1, -1, -1):
        G = gamma * G + rewards[t]
        if not (states[t] == states[:t]).all(axis=1).any():
            idx = tuple(states[t]) + (actions[t],)
            M[idx] += 1
            Q[idx] += 1 / M[idx] * (G - Q[idx])
    return M, Q
