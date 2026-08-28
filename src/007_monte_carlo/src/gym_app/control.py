import numpy as np
from numpy.typing import NDArray


def b(state: NDArray, Q: NDArray, epsilon: float) -> int:
    """
    Behavior Policy
    Epsilon-greedy w.r.t state-action values (specialized for 2 actions)
    """
    if np.random.random() < epsilon:
        return np.random.randint(2)
    return pi(state, Q)


def pi(state: NDArray, Q: NDArray) -> int:
    """
    Target Policy
    Greedy
    """
    q0, q1 = Q[tuple(state)]
    if q0 > q1:
        return 0
    elif q1 > q0:
        return 1
    return np.random.randint(2)
