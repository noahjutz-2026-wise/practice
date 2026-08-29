import numpy as np
from numpy.typing import NDArray


def pi(state: NDArray, Q: NDArray, epsilon: float) -> int:
    if epsilon < 1 and np.random.random() < epsilon:
        return np.random.randint(2)
    return _greedy(state, Q)


def _greedy(state: NDArray, Q: NDArray) -> int:
    q0, q1 = Q[tuple(state)]
    if q0 > q1:
        return 0
    elif q1 > q0:
        return 1
    return np.random.randint(2)
