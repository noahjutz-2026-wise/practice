import numpy as np
from numpy.typing import NDArray

"""
First-Visit MC prediction
Given one episode
"""


def monte_carlo(
    rewards: NDArray, states: NDArray, gamma: float, M: NDArray, V: NDArray
) -> tuple[NDArray, NDArray]:
    G = 0
    for t in range(len(rewards) - 1, -1, -1):
        G = gamma * G + rewards[t]
        if not (states[t] == states[:t]).all(axis=1).any():
            o = tuple(states[t])
            M[o] += 1
            V[o] += 1 / M[o] * (G - V[o])
    return M, V
